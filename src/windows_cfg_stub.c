#if defined(_WIN64) && defined(__x86_64__)
__attribute__((naked)) void __guard_dispatch_icall_dummy(void) {
    __asm__("jmp *%rax");
}
#endif
