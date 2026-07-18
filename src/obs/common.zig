pub const MetricKind = enum {
    gauge,
    counter,
};

pub const MetricUnit = enum {
    count,
    bytes,
    nanoseconds,
};

pub const Mechanism = enum {
    diagnostic,
    telemetry,
};

pub const TelemetryChannel = enum {
    traces,
    logs,
    metrics,
};

pub const Domain = enum {
    runtime,
    compute,
    memory,
    support,
};
