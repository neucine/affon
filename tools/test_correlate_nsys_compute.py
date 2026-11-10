import importlib.util
import sqlite3
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("correlate-nsys-compute.py")
SPEC = importlib.util.spec_from_file_location("correlate_nsys_compute", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CorrelationTest(unittest.TestCase):
    def test_driver_api_joins_kernel_and_keeps_unmatched_activity(self):
        db = sqlite3.connect(":memory:")
        db.executescript("""
            create table StringIds(id integer, value text);
            create table NVTX_EVENTS(start integer, end integer, text text, textId integer, globalTid integer);
            create table CUPTI_ACTIVITY_KIND_DRIVER(start integer, end integer, globalTid integer, correlationId integer, nameId integer);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer, end integer, globalTid integer, correlationId integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL(start integer, end integer, correlationId integer, bytes integer);
            create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer, end integer, correlationId integer, bytes integer);
            create table CUPTI_ACTIVITY_KIND_MEMSET(start integer, end integer, correlationId integer, bytes integer);
            create table CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(start integer, end integer, globalTid integer, correlationId integer, syncType integer);
            insert into StringIds values (1, 'affon run=4 executable=8 step=2 invocation=11');
            insert into StringIds values (2, 'cuLaunchKernel');
            insert into StringIds values (3, 'cuStreamSynchronize');
            insert into StringIds values (4, 'cuMemcpyDtoH');
            insert into StringIds values (5, 'cuMemsetD32Async');
            insert into NVTX_EVENTS values (100, 200, null, 1, 7);
            insert into NVTX_EVENTS values (210, 350, 'affon run=4 executable=8 step=3 invocation=12', null, 7);
            insert into NVTX_EVENTS values (400, 500, 'affon readback tensor=5 storage=7 operation=to_bytes', null, 7);
            insert into NVTX_EVENTS values (322, 336, 'cublasSgemm_v2', null, 7);
            insert into CUPTI_ACTIVITY_KIND_DRIVER values (120, 130, 7, 91, 2);
            insert into CUPTI_ACTIVITY_KIND_RUNTIME values (120, 130, 7, 91);
            insert into CUPTI_ACTIVITY_KIND_DRIVER values (325, 335, 7, 92, 3);
            insert into CUPTI_ACTIVITY_KIND_DRIVER values (410, 420, 7, 94, 4);
            insert into CUPTI_ACTIVITY_KIND_DRIVER values (425, 430, 7, 95, 5);
            insert into CUPTI_ACTIVITY_KIND_DRIVER values (120, 130, 8, 93, 2);
            insert into CUPTI_ACTIVITY_KIND_KERNEL values (300, 320, 91, 0);
            insert into CUPTI_ACTIVITY_KIND_KERNEL values (340, 360, 92, 0);
            insert into CUPTI_ACTIVITY_KIND_KERNEL values (370, 390, 93, 0);
            insert into CUPTI_ACTIVITY_KIND_MEMCPY values (430, 460, 94, 8192);
            insert into CUPTI_ACTIVITY_KIND_MEMSET values (430, 440, 95, 4096);
            insert into CUPTI_ACTIVITY_KIND_SYNCHRONIZATION values (326, 334, 7, 92, 3);
        """)
        result = MODULE.analyze(db)
        self.assertEqual(result["total_device_activities"], 5)
        self.assertEqual(result["unmatched_device_activities"], 1)
        self.assertEqual(
            result["unmatched_device_activity_details"],
            [
                {
                    "start_ns": 370,
                    "end_ns": 390,
                    "kind": "kernel",
                    "correlation_id": 93,
                    "bytes": 0,
                }
            ],
        )
        self.assertEqual(result["runs"][0]["device_gap_ns"], 20)
        self.assertEqual(result["runs"][0]["cuda_api_count"], 2)
        self.assertEqual(
            result["runs"][0]["classification"],
            {
                "gap_count": 1,
                "duration_ns": 20,
                "synchronization_ns": 8,
                "cublas_api_ns": 6,
                "cuda_api_ns": 0,
                "host_delay_ns": 6,
            },
        )
        self.assertEqual(result["runs"][0]["synchronization_apis"][0]["name"], "cuStreamSynchronize")
        self.assertEqual(result["runs"][0]["synchronization_apis"][0]["coverage_ns"], 8)
        transitions = result["runs"][0]["gap_transitions"]
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["gap_count"], 1)
        self.assertEqual(transitions[0]["cublas_api_ns"], 6)
        self.assertEqual(transitions[0]["host_delay_ns"], 6)
        self.assertEqual(
            [point["step_index"] for point in transitions[0]["preceding"]],
            [2],
        )
        self.assertEqual(
            [point["step_index"] for point in transitions[0]["following"]],
            [3],
        )
        self.assertEqual(result["total_synchronization_activities"], 1)
        self.assertEqual(result["unmatched_synchronization_activities"], 0)
        self.assertEqual(result["total_cublas_activities"], 1)
        self.assertEqual(result["unmatched_cublas_activities"], 0)
        self.assertEqual(result["runs"][0]["cublas_api_count"], 1)
        self.assertEqual(result["runs"][0]["cublas_api_ns"], 14)
        self.assertEqual(result["runs"][0]["cublas_apis"][0]["name"], "cublasSgemm_v2")
        self.assertEqual(result["runs"][0]["cublas_apis"][0]["coverage_ns"], 14)
        self.assertEqual([step["kernel_count"] for step in result["runs"][0]["steps"]], [1, 1])
        self.assertEqual(result["runs"][0]["invocation_ids"], [11, 12])
        self.assertEqual(result["runs"][0]["run_id_text"], "4")
        self.assertEqual(result["runs"][0]["executable_id_text"], "8")
        self.assertEqual(result["runs"][0]["invocation_ids_text"], ["11", "12"])
        self.assertEqual(
            [step["invocation_ids"] for step in result["runs"][0]["steps"]],
            [[11], [12]],
        )
        self.assertEqual(
            [step["invocation_ids_text"] for step in result["runs"][0]["steps"]],
            [["11"], ["12"]],
        )
        self.assertEqual(len(result["readbacks"]), 1)
        self.assertEqual(result["readbacks"][0]["operation"], "to_bytes")
        self.assertEqual(result["readbacks"][0]["storage_id_text"], "7")
        self.assertEqual(result["readbacks"][0]["memcpy_count"], 1)
        self.assertEqual(result["readbacks"][0]["memcpy_bytes"], 8192)
        self.assertEqual(result["readbacks"][0]["memset_count"], 1)
        self.assertEqual(result["readbacks"][0]["memset_bytes"], 4096)
        self.assertEqual(result["readbacks"][0]["device_active_ns"], 30)


if __name__ == "__main__":
    unittest.main()
