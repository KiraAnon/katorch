/* 查询手机真实 L1 内存大小，用于修正 PC 端编译器的硬件参数 */
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

int main(void) {
    const char* p = "/data/data/com.termux/files/usr/lib/libneuron_adapter_mgvi.so";
    void* h = dlopen(p, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("{\"err\":\"dlopen\",\"msg\":\"%s\"}\n", dlerror()); return 1; }

    int (*fn)(uint32_t*) = (int (*)(uint32_t*))dlsym(h, "Neuron_getL1MemorySizeKb");
    if (!fn) { printf("{\"err\":\"dlsym\"}\n"); return 2; }

    uint32_t kb = 0;
    int rc = fn(&kb);
    printf("{\"rc\":%d,\"l1_kb\":%u}\n", rc, kb);
    return 0;
}
