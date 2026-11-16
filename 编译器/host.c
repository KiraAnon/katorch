// neuron_rt host —— 绕过 NNAPI，直接驱动 MediaTek Neuron Adapter / MDLA
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#define LIBDIR "/data/data/com.termux/files/usr/lib/"
#define MAXOP 65536

typedef struct { int32_t type; uint32_t nd; const uint32_t* dims; float scale; int32_t zp; } OT;

static void *h_adp, *h_dev_mdla;
#define MAXSLOT 16
static void *g_models[MAXSLOT];
static void *g_comps[MAXSLOT];
static int g_nslots = 0;
static int g_slot = 0;
static int g_nslots_restored[MAXSLOT];
static char g_cache_dir[512] = "";
static char g_cache_key[600] = "";
static uint32_t g_dims[MAXOP][8];
static int g_ndims[MAXOP];
static long g_bytes[MAXOP];
#define MAXIO 1024
static uint32_t g_in_idx[MAXSLOT][MAXIO], g_out_idx[MAXSLOT][MAXIO];
static int g_n_in[MAXSLOT] = {0}, g_n_out[MAXSLOT] = {0};
// restore 路径下由 compilation 查询到的 I/O 尺寸（不依赖 g.txt）
static size_t g_pad_in[MAXSLOT][MAXIO], g_pad_out[MAXSLOT][MAXIO];
// restore 用的 blob buffer 必须活到进程结束：adapter 可能是零拷贝引用它
static void* g_rst_buf[MAXSLOT];
// 常驻内存的 blob 池：bload 一次性读入，map 时只做映射，不碰磁盘
#define MAXBLOB 64
static void* g_blob_ptr[MAXBLOB];
static long  g_blob_len[MAXBLOB];
static int   g_nblob = 0;
static int   g_mapped = 0;          // 当前是否有 map 出来的块
static void* g_wbuf[MAXOP];   // 权重的常量 buffer 必须活到 compile 结束（adapter 不拷贝）
// per-channel 量化：通道维、通道数、scale 数组都必须活到 compile 结束
#define MAXQCH 1024
typedef struct { uint32_t channelDim; uint32_t scaleCount; const float* scales; } SPCQ;
static SPCQ g_qch[MAXQCH];
static float* g_qch_scales[MAXQCH];
static int   g_qch_opid[MAXQCH];
static int   g_nqch = 0;
#define MAXOPS 16384
static int g_n_op_count[MAXSLOT];
static int g_op_code[MAXSLOT][MAXOPS];

static int (*M_create)(void**);
static int (*M_addOp)(void*, const OT*);
static int (*M_setVal)(void*, int32_t, const void*, size_t);
static int (*M_addOpn)(void*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
static int (*M_ident)(void*, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
static int (*M_fin)(void*);
static int (*C_cr)(void*, void* const*, uint32_t, void**);
static int (*C_cr_nd)(void*, void**);  // NeuronCompilation_create (无设备)
static int (*C_fin)(void*);
static int (*C_gopt)(void*, const char**);  // getOptimizationString
static int (*C_cache)(void*, const char*, const uint8_t*);
static int (*E_cr)(void*, void**);
static int (*E_in)(void*, int32_t, const OT*, const void*, size_t);
static int (*E_out)(void*, int32_t, const OT*, void*, size_t);
static int (*E_run)(void*);
static int (*C_gsz)(void*, size_t*);
static int (*C_str)(void*, void*, size_t);
// 官方签名：model 和 compilation 都是输出指针（NeuronModel**, NeuronCompilation**）
static int (*M_rst)(void**, void**, const void*, size_t);
static int (*C_isz)(void*, int32_t, size_t*);   // getInputPaddedSize
static int (*C_osz)(void*, int32_t, size_t*);   // getOutputPaddedSize
static int (*M_free)(void*);                    // NeuronModel_free
static int (*M_qch)(void*, int32_t, const SPCQ*);  // setOperandSymmPerChannelQuantParams
static int (*M_supd)(const void*, const void* const*, uint32_t, bool*);  // getSupportedOperationsForDevices
static int (*M_dbgpath)(void*, const char*);   // NeuronDebug_setReportPath
static int (*C_crdbg)(void*, void**);          // NeuronCompilation_createForDebug
static int g_usedbg = 0;
static int (*C_free)(void*);                    // NeuronCompilation_free
static int (*M_sup)(void*, uint32_t*, uint32_t*);
static int (*M_relax)(void*, int);
static long g_blob_size;
static int g_relax = 1;
static int g_nodev = 0;    // 1 = 不指定设备（NeuronCompilation_create）    // 1 = 允许 fp32 降 fp16（默认，兼容老图）

static void reply(const char* s){ printf("%s\n", s); fflush(stdout); }

static int elem_size(int type){
    switch (type){
        case 1: case 4: return 4;      // INT32 / TENSOR_INT32
        case 5: return 1;              // QUANT8_ASYMM
        case 11: case 13: case 14: return 1;   // QUANT8_SYMM_PER_CHANNEL / SYMM / ASYMM_SIGNED
        case 7: case 12: return 2;     // QUANT16_SYMM / QUANT16_ASYMM
        case 8: return 2;              // TENSOR_FLOAT16
        default: return 4;             // FLOAT32 / 其他
    }
}



static int load_adapter(void){
    dlopen(LIBDIR "libapu_mdw.so",       RTLD_NOW|RTLD_GLOBAL);
    dlopen(LIBDIR "libapu_mdw_batch.so", RTLD_NOW|RTLD_GLOBAL);
    dlopen(LIBDIR "libapusys.so",        RTLD_NOW|RTLD_GLOBAL);
    h_adp = dlopen(LIBDIR "libneuron_adapter_mgvi.so", RTLD_NOW|RTLD_GLOBAL);
    if (!h_adp) return 0;
    int (*gdc)(uint32_t*) = (int(*)(uint32_t*))dlsym(h_adp,"Neuron_getDeviceCount");
    int (*gd)(uint32_t, void**) = (int(*)(uint32_t, void**))dlsym(h_adp,"Neuron_getDevice");
    int (*gn)(void*, const char**) = (int(*)(void*, const char**))dlsym(h_adp,"NeuronDevice_getName");
    M_create = (int(*)(void**))dlsym(h_adp,"NeuronModel_create");
    M_addOp  = (int(*)(void*, const OT*))dlsym(h_adp,"NeuronModel_addOperand");
    M_setVal = (int(*)(void*, int32_t, const void*, size_t))dlsym(h_adp,"NeuronModel_setOperandValue");
    M_addOpn = (int(*)(void*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*))dlsym(h_adp,"NeuronModel_addOperation");
    M_ident  = (int(*)(void*, uint32_t, const uint32_t*, uint32_t, const uint32_t*))dlsym(h_adp,"NeuronModel_identifyInputsAndOutputs");
    M_fin    = (int(*)(void*))dlsym(h_adp,"NeuronModel_finish");
    C_cr     = (int(*)(void*, void* const*, uint32_t, void**))dlsym(h_adp,"NeuronCompilation_createForDevices");
    C_cr_nd = (int(*)(void*, void**))dlsym(h_adp,"NeuronCompilation_create");
    C_fin    = (int(*)(void*))dlsym(h_adp,"NeuronCompilation_finish");
    C_gopt   = (int(*)(void*, const char**))dlsym(h_adp,"NeuronCompilation_getOptimizationString");
    C_cache  = (int(*)(void*, const char*, const uint8_t*))dlsym(h_adp,"NeuronCompilation_setCaching");
    E_cr     = (int(*)(void*, void**))dlsym(h_adp,"NeuronExecution_create");
    E_in     = (int(*)(void*, int32_t, const OT*, const void*, size_t))dlsym(h_adp,"NeuronExecution_setInput");
    E_out    = (int(*)(void*, int32_t, const OT*, void*, size_t))dlsym(h_adp,"NeuronExecution_setOutput");
    E_run    = (int(*)(void*))dlsym(h_adp,"NeuronExecution_compute");
    C_gsz    = (int(*)(void*, size_t*))dlsym(h_adp,"NeuronCompilation_getCompiledNetworkSize");
    C_str    = (int(*)(void*, void*, size_t))dlsym(h_adp,"NeuronCompilation_storeCompiledNetwork");
    M_rst    = (int(*)(void**, void**, const void*, size_t))dlsym(h_adp,"NeuronModel_restoreFromCompiledNetwork");
    C_isz    = (int(*)(void*, int32_t, size_t*))dlsym(h_adp,"NeuronCompilation_getInputPaddedSize");
    C_osz    = (int(*)(void*, int32_t, size_t*))dlsym(h_adp,"NeuronCompilation_getOutputPaddedSize");
    M_free   = (int(*)(void*))dlsym(h_adp,"NeuronModel_free");
    M_qch    = (int(*)(void*, int32_t, const SPCQ*))dlsym(h_adp,"NeuronModel_setOperandSymmPerChannelQuantParams");
    M_supd   = (int(*)(const void*, const void* const*, uint32_t, bool*))dlsym(h_adp,"NeuronModel_getSupportedOperationsForDevices");
    M_dbgpath= (int(*)(void*, const char*))dlsym(h_adp,"NeuronDebug_setReportPath");
    C_crdbg  = (int(*)(void*, void**))dlsym(h_adp,"NeuronCompilation_createForDebug");
    C_free   = (int(*)(void*))dlsym(h_adp,"NeuronCompilation_free");
    M_sup    = (int(*)(void*, uint32_t*, uint32_t*))dlsym(h_adp,"NeuronModel_getSupportedOperations");
    M_relax = (int(*)(void*, int))dlsym(h_adp,"NeuronModel_relaxComputationFloat32toFloat16");
    uint32_t n = 0; gdc(&n);
    for (uint32_t i = 0; i < n; i++){
        void* d = 0; const char* nm = 0;
        gd(i, &d); gn(d, &nm);
        if (nm && strcmp(nm, "mtk-mdla") == 0) h_dev_mdla = d;
    }
    return h_dev_mdla != 0;
}

static long file_size(const char* p){
    FILE* f = fopen(p, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f); return n;
}

static int build_from_file(const char* path){
    g_nqch = 0;   // 每次 build 重置 per-channel 队列
    if (g_slot >= 0 && g_slot < MAXSLOT) g_n_op_count[g_slot] = 0;
    FILE* f = fopen(path, "r");
    if (!f) return -101;
    char line[4096];
    static uint32_t in_idx[MAXIO], out_idx[MAXIO]; int n_in = 0, n_out = 0, n_op = 0;
    void* m = 0;
    if (M_create(&m)) { fclose(f); return -102; }
    while (fgets(line, sizeof line, f)){
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == 0) continue;
        if (!strncmp(p, "OPD", 3)){
            int type, nd, zp; float scale;
            p += 3;
            int id = (int)strtol(p, &p, 10);
            type = (int)strtol(p, &p, 10);
            nd = (int)strtol(p, &p, 10);
            long nelem = 1;
            for (int i = 0; i < nd; i++){ g_dims[id][i] = (uint32_t)strtol(p, &p, 10); nelem *= g_dims[id][i]; }
            g_ndims[id] = nd;
            g_bytes[id] = nelem * elem_size(type);
            scale = strtof(p, &p);
            zp = (int)strtol(p, &p, 10);
            while (*p == ' ') p++;
            OT ot = { type, (uint32_t)nd, g_dims[id], scale, zp };
            int rc = M_addOp(m, &ot);
            if (rc) { fclose(f); return -200 - rc; }
            if (*p == 'w'){
                p++; while (*p == ' ') p++;
                char* e2 = p; while (*e2 && *e2 != '\n' && *e2 != '\r') e2++; *e2 = 0;
                // 两种写法：
                //   w <path>            整个文件就是一个常量
                //   w <path> <off> <len>  从打包文件里取一段（小常量合并用，
                //                         避免几千次 setOperandValue 各建一个 IOVA buffer）
                long sz = -1, off = 0;
                char* sp = strchr(p, ' ');
                if (sp){
                    *sp = 0;
                    char* q = sp + 1;
                    off = strtol(q, &q, 10);
                    sz = strtol(q, &q, 10);
                } else {
                    sz = file_size(p);
                }
                if (sz < 0) { fclose(f); return -104; }
                void* buf = malloc(sz);
                FILE* wf = fopen(p, "rb"); if (!wf) { free(buf); fclose(f); return -105; }
                if (off) fseek(wf, off, SEEK_SET);
                if (fread(buf, 1, sz, wf) != (size_t)sz) { fclose(wf); free(buf); fclose(f); return -106; }
                fclose(wf);
                rc = M_setVal(m, id, buf, (size_t)sz);
                if (rc) { free(buf); fclose(f); return -300 - rc; }
                g_wbuf[id] = buf;   // 保留到 compile 之后
            } else if (*p == 'a'){
                p++; int32_t v = (int32_t)strtol(p, &p, 10);
                rc = M_setVal(m, id, &v, 4);
                if (rc) { fclose(f); return -310 - rc; }
            }
        } else if (!strncmp(p, "IO", 2)){
            p += 2;
            n_in = (int)strtol(p, &p, 10);
            for (int i = 0; i < n_in; i++) in_idx[i] = (uint32_t)strtol(p, &p, 10);
            n_out = (int)strtol(p, &p, 10);
            for (int i = 0; i < n_out; i++) out_idx[i] = (uint32_t)strtol(p, &p, 10);
        } else if (!strncmp(p, "QCH", 3)){
            // QCH <opid> <channelDim> <count> <scales_f32_path>
            p += 3;
            int oid = (int)strtol(p, &p, 10);
            int cdim = (int)strtol(p, &p, 10);
            int cnt = (int)strtol(p, &p, 10);
            while (*p == ' ') p++;
            char* e = p; while (*e && *e != '\n' && *e != '\r') e++; *e = 0;
            if (g_nqch < MAXQCH && cnt > 0){
                float* sc = (float*)malloc(sizeof(float) * (size_t)cnt);
                FILE* sf = fopen(p, "rb");
                if (!sc || !sf) { if (sf) fclose(sf); free(sc); return -115; }
                if (fread(sc, sizeof(float), (size_t)cnt, sf) != (size_t)cnt){
                    fclose(sf); free(sc); return -116;
                }
                fclose(sf);
                g_qch_scales[g_nqch] = sc;
                g_qch[g_nqch].channelDim = (uint32_t)cdim;
                g_qch[g_nqch].scaleCount = (uint32_t)cnt;
                g_qch[g_nqch].scales = sc;
                g_qch_opid[g_nqch] = oid;
                g_nqch++;
            }
        } else if (!strncmp(p, "OP", 2)){
            p += 2;
            int op = (int)strtol(p, &p, 10);
            int ni = (int)strtol(p, &p, 10);
            uint32_t ii[32];
            for (int i = 0; i < ni; i++) ii[i] = (uint32_t)strtol(p, &p, 10);
            int no = (int)strtol(p, &p, 10);
            uint32_t oo[32];
            for (int i = 0; i < no; i++) oo[i] = (uint32_t)strtol(p, &p, 10);
            int rc = M_addOpn(m, (int32_t)op, (uint32_t)ni, ii, (uint32_t)no, oo);
            if (g_slot >= 0 && g_slot < MAXSLOT && g_n_op_count[g_slot] < MAXOPS)
                g_op_code[g_slot][g_n_op_count[g_slot]] = op;
            g_n_op_count[g_slot]++;
            if (rc) { fclose(f); return -400 - rc; }
            n_op++;
        } else if (!strncmp(p, "END", 3)){
            break;
        }
    }
    fclose(f);
    if (n_in == 0 || n_out == 0){
        fprintf(stderr, "[BUILD] 解析失败: n_in=%d n_out=%d n_op=%d nqch=%d\n",
                n_in, n_out, n_op, g_nqch);
        return -107;
    }
    if (M_relax) M_relax(m, g_relax);
    // per-channel 量化参数必须在 addOperand 之后、finish 之前设置
    if (M_qch){
        for (int i = 0; i < g_nqch; i++){
            if (M_qch(m, g_qch_opid[i], &g_qch[i])) return -117;
        }
    }
    if (M_ident(m, (uint32_t)n_in, in_idx, (uint32_t)n_out, out_idx)) return -108;
    g_n_in[g_slot] = n_in; g_n_out[g_slot] = n_out;
    memcpy(g_in_idx[g_slot], in_idx, sizeof(uint32_t)*n_in);
    memcpy(g_out_idx[g_slot], out_idx, sizeof(uint32_t)*n_out);
    if (M_fin(m)) return -109;
    g_models[g_slot] = m;
    return n_op;
}

static int compile_model(void){
    if (!g_models[g_slot]) return -110;
    // 即便 restore 过也必须建 compilation（run 需要它），
    // 只是 finish 会复用恢复出来的产物，不再真编译
    void* comp = 0;
    void* devs[1] = { h_dev_mdla };
    int rc;
    if (g_usedbg && C_crdbg){
        rc = C_crdbg(g_models[g_slot], &comp);
    } else if (g_nodev && C_cr_nd){
        rc = C_cr_nd(g_models[g_slot], &comp);
    } else {
        /* DUMP_DEV: 打印 Device 结构的 48 字节 */
        if (devs[0]) {
            uint64_t* p = (uint64_t*)devs[0];
            fprintf(stderr, "[DEV] ptr=%p  raw:", devs[0]);
            for (int k = 0; k < 6; k++) fprintf(stderr, " [+%d]=0x%016llx", k*8, (unsigned long long)p[k]);
            fprintf(stderr, "\n");
        }
        rc = C_cr(g_models[g_slot], devs, 1, &comp);
    }
    if (rc) return -500 - rc;
    if (C_cache && g_cache_dir[0]){
        uint8_t token[8];
        uint64_t hh = 1469598103934665603ULL;
        for (const char* q = g_cache_key; *q; q++){ hh ^= (uint8_t)*q; hh *= 1099511628211ULL; }
        memcpy(token, &hh, 8);
        int crc = C_cache(comp, g_cache_dir, token);
        if (crc) return -520 - crc;
    }
    if (C_gopt){ const char* osv=NULL; int orc=C_gopt(comp,&osv);
        fprintf(stderr,"[OPT] rc=%d str=%s\n", orc, osv?osv:"(null)"); }
    rc = C_fin(comp);
    if (rc){
        // restore 过的图 C_fin 会返回 BAD_DATA（它已经是编译好的），
        // 但 comp 仍然可以用来创建 execution
        if (g_nslots_restored[g_slot]){
            g_comps[g_slot] = comp;
            return 0;
        }
        return -510 - rc;
    }
    for (int i = 0; i < MAXOP; i++) if (g_wbuf[i]) { free(g_wbuf[i]); g_wbuf[i] = 0; }
    g_comps[g_slot] = comp;
    return 0;
}

// 导出编译产物。adapter 没有 getCompiledNetwork，只有 storeCompiledNetwork(compil, path)
static int export_blob(const char* path){
    if (!g_comps[g_slot]) return -700;
    if (!C_gsz) return -701;
    if (!C_str) return -702;
    size_t sz = 0;
    int rc = C_gsz(g_comps[g_slot], &sz);
    if (rc) return -711 - rc;
    if (!sz) return -712;
    void* buf = malloc(sz);
    if (!buf) return -713;
    rc = C_str(g_comps[g_slot], buf, sz);
    if (rc){ free(buf); return -721 - rc; }
    FILE* f = fopen(path, "wb");
    if (!f){ free(buf); return -714; }
    size_t w = fwrite(buf, 1, sz, f);
    fclose(f); free(buf);
    if (w != sz) return -715;
    g_blob_size = (long)sz;
    return 0;
}

// 从已经在内存里的 blob 恢复（不碰磁盘）。buf 由调用方持有并保证存活。
static int restore_from_buf(void* buf, long sz, int tag){
    if (!M_rst) return -730;
    struct timespec tp0, tp1;
    clock_gettime(CLOCK_MONOTONIC, &tp0);
    void* model = 0;
    void* comp  = 0;
    int rc = M_rst(&model, &comp, buf, (size_t)sz);
    clock_gettime(CLOCK_MONOTONIC, &tp1);
    double ms_map = (tp1.tv_sec - tp0.tv_sec) * 1e3 + (tp1.tv_nsec - tp0.tv_nsec) / 1e6;
    fprintf(stderr, "[MAP] idx=%d size=%.1fMB map=%.0fms rc=%d\n", tag, sz / 1e6, ms_map, rc);
    if (rc) return -724 - rc;
    if (!comp) return -750;
    g_rst_buf[g_slot] = buf;
    g_models[g_slot] = model;
    g_comps[g_slot]  = comp;
    g_nslots_restored[g_slot] = 1;

    // 从 compilation 直接查询 I/O 尺寸，不依赖 g.txt
    int n_in = 0, n_out = 0;
    if (C_isz){
        for (int i = 0; i < MAXIO; i++){
            size_t s = 0;
            if (C_isz(comp, (int32_t)i, &s)) break;
            g_pad_in[g_slot][i] = s;
            n_in++;
        }
    }
    if (C_osz){
        for (int i = 0; i < MAXIO; i++){
            size_t s = 0;
            if (C_osz(comp, (int32_t)i, &s)) break;
            g_pad_out[g_slot][i] = s;
            n_out++;
        }
    }
    g_n_in[g_slot]  = n_in;
    g_n_out[g_slot] = n_out;
    return 0;
}

// 从磁盘上的 blob 文件恢复（会自动把 buffer 留在 g_rst_buf 里）
static int restore_blob(const char* path){
    if (!M_rst) return -730;
    long sz = file_size(path);
    if (sz <= 0) return -721;
    void* buf = malloc((size_t)sz);
    if (!buf) return -722;
    FILE* f = fopen(path, "rb");
    if (!f || fread(buf, 1, (size_t)sz, f) != (size_t)sz){ if (f) fclose(f); free(buf); return -723; }
    fclose(f);
    int rc = restore_from_buf(buf, sz, -1);
    if (rc) free(buf);
    return rc;
}

// 释放某个 slot 的映射，回收 IOVA
static int release_slot(int s){
    if (s < 0 || s >= MAXSLOT) return -760;
    int rf = -1, cf = -1;
    if (C_free && g_comps[s]){ cf = (C_free(g_comps[s]), 0); g_comps[s] = 0; }
    if (M_free && g_models[s]){ rf = (M_free(g_models[s]), 0); g_models[s] = 0; }
    // 注意：g_rst_buf 指向常驻 blob 时不释放（属于 blob 池）
    g_rst_buf[s] = 0;
    g_nslots_restored[s] = 0;
    g_n_in[s] = g_n_out[s] = 0;
    (void)rf;
    return 0;
}


// ---------------- 链式执行 ----------------
// 目的：分块跑同一个模型时，块之间的中间结果留在内存，不落盘。
// 单块 I/O 实测占 86.56ms 的约 65%，链式执行能把它降到接近 0。
#define MAXPOOL 16
static void* g_pool[MAXPOOL];
static long  g_pool_sz[MAXPOOL];

static void pool_clear(void){
    for (int i = 0; i < MAXPOOL; i++){ if (g_pool[i]){ free(g_pool[i]); g_pool[i] = 0; } g_pool_sz[i] = 0; }
}

// 读一个输入：文件名 或 @N（内存池）
static void* chain_get(const char* tok, long* sz_out){
    if (tok[0] == '@'){
        int n = atoi(tok + 1);
        if (n < 0 || n >= MAXPOOL || !g_pool[n]) return 0;
        *sz_out = g_pool_sz[n];
        return g_pool[n];
    }
    long sz = file_size(tok);
    if (sz < 0) return 0;
    void* d = malloc(sz);
    FILE* f = fopen(tok, "rb");
    if (!f || fread(d, 1, sz, f) != (size_t)sz){ if (f) fclose(f); free(d); return 0; }
    fclose(f);
    *sz_out = sz;
    return d;
}

// 写一个输出：文件名 或 @N（内存池）
static int chain_put(const char* tok, const void* data, long sz){
    if (tok[0] == '@'){
        int n = atoi(tok + 1);
        if (n < 0 || n >= MAXPOOL) return -1;
        if (g_pool[n]) free(g_pool[n]);
        g_pool[n] = malloc(sz);
        memcpy(g_pool[n], data, sz);
        g_pool_sz[n] = sz;
        return 0;
    }
    FILE* f = fopen(tok, "wb");
    if (!f) return -1;
    size_t w = fwrite(data, 1, sz, f);
    fclose(f);
    return w == (size_t)sz ? 0 : -1;
}

// spec 形如 "0:a.bin,b.bin,@1;1:@1,b.bin,@2;2:@2,b.bin,out.bin"
// 每段 = "槽号:输入1,输入2,...,输出"
static int chain_run(const char* spec){
    // 注意：不清空 g_pool —— 跨多次 chain 命令要用 @N 传递中间结果。
    // 需要释放时发 {"cmd":"poolclear"}。
    char buf[8192]; strncpy(buf, spec, sizeof buf - 1); buf[sizeof buf - 1] = 0;
    char* seg = buf;
    int step = 0, rc = 0;
    while (seg && *seg){
        char* semi = strchr(seg, ';');
        if (semi) *semi = 0;
        char* colon = strchr(seg, ':');
        if (!colon){ rc = -801; break; }
        *colon = 0;
        int slot = atoi(seg);
        if (slot < 0 || slot >= g_nslots){ rc = -802; break; }
        g_slot = slot;

        // 拆输入和输出
        char* last_comma = strrchr(colon + 1, ',');
        if (!last_comma){ rc = -803; break; }
        *last_comma = 0;
        char* outp = last_comma + 1;

        void* ex = 0;
        if (E_cr(g_comps[slot], &ex)){ rc = -810; break; }
        void* keep[MAXIO]; int nk = 0;
        char* inp = colon + 1;
        int idx = 0;
        while (inp && *inp && idx < MAXIO){
            char* comma = strchr(inp, ',');
            if (comma) *comma = 0;
            if (*inp){
                long sz = 0;
                void* d = chain_get(inp, &sz);
                if (!d){ rc = -811; break; }
                int r = E_in(ex, idx, 0, d, (size_t)sz);
                if (r){ free(d); rc = -812 - r; break; }
                keep[nk++] = d;
                idx++;
            }
            inp = comma ? comma + 1 : 0;
        }
        if (rc) break;

        // 输出只有一个
        int opid = (int)g_out_idx[slot][0];
        if (opid < 0 || opid >= MAXOP || g_bytes[opid] <= 0){ rc = -813; break; }
        long osz = g_bytes[opid];
        void* obuf = malloc((size_t)osz);
        memset(obuf, 0, (size_t)osz);
        int r = E_out(ex, 0, 0, obuf, (size_t)osz);
        if (r){ free(obuf); rc = -814 - r; break; }
        r = E_run(ex);
        for (int i = 0; i < nk; i++) free(keep[i]);
        if (r){ free(obuf); rc = -815 - r; break; }
        if (chain_put(outp, obuf, osz)){ free(obuf); rc = -816; break; }
        free(obuf);

        step++;
        seg = semi ? semi + 1 : 0;
    }
    return rc ? rc : step;
}

// spec: "<n_in> <idx> <path> ... <n_out> <idx> <path> ..."
static const char* g_dump_dir = 0;   // 非空时：所有输出写到 <dir>/o<n>.bin（标定用）

static int run_spec(const char* spec){
    char buf[4096]; strncpy(buf, spec, sizeof buf - 1); buf[sizeof buf - 1] = 0;
    char* p = buf;
    int n_in = (int)strtol(p, &p, 10);
    void* ex = 0;
    // 普通路径用 compilation；restore 路径下 compilation 就是可执行句柄
    const int rst = g_nslots_restored[g_slot];
    void* target = g_comps[g_slot];
    if (!target) return -599;
    if (E_cr(target, &ex)) return -600;
    void* keep[MAXIO];
    for (int i = 0; i < n_in; i++){
        int slot = (int)strtol(p, &p, 10);
        while (*p == ' ') p++;
        char path[512]; int k = 0;
        while (*p && *p != ' ' && k < 511) path[k++] = *p++;
        path[k] = 0;
        if (slot < 0 || slot >= g_n_in[g_slot]) return -607;
        long sz = file_size(path);
        if (sz < 0) return -601;
        // restore 路径：以 compilation 声明的尺寸为准；文件更小则补零
        size_t need = (size_t)sz;
        if (rst && g_pad_in[g_slot][slot] > 0) need = g_pad_in[g_slot][slot];
        void* data = malloc(need);
        if (!data) return -603;
        memset(data, 0, need);
        FILE* f = fopen(path, "rb");
        size_t want = (size_t)sz < need ? (size_t)sz : need;
        if (!f || fread(data, 1, want, f) != want) { if (f) fclose(f); free(data); return -602; }
        fclose(f);
        int rc = E_in(ex, slot, 0, data, need);
        if (rc) { free(data); return -610 - rc; }
        keep[i] = data;   // 必须保持到 compute 结束：adapter 可能是零拷贝引用
    }
    int n_out = (int)strtol(p, &p, 10);
    void* outs[MAXIO]; static long osz[MAXIO]; static int oidx[MAXIO]; static char opath[MAXIO][512];
    for (int i = 0; i < n_out; i++){
        int slot = (int)strtol(p, &p, 10);
        while (*p == ' ') p++;
        int k = 0;
        while (*p && *p != ' ' && k < 511) opath[i][k++] = *p++;
        opath[i][k] = 0;
        if (slot < 0 || slot >= g_n_out[g_slot]) return -605;
        if (rst && g_pad_out[g_slot][slot] > 0){
            osz[i] = (long)g_pad_out[g_slot][slot];
        } else {
            int opid = (int)g_out_idx[g_slot][slot];
            if (opid < 0 || opid >= MAXOP || g_bytes[opid] <= 0) return -606;
            osz[i] = g_bytes[opid];
        }
        outs[i] = malloc(osz[i]);
        memset(outs[i], 0, osz[i]);
        int rc = E_out(ex, slot, 0, outs[i], (size_t)osz[i]);   // index 是模型输出序号
        if (rc) return -620 - rc;
    }
    int rc = E_run(ex);
    for (int i = 0; i < n_in; i++) free(keep[i]);
    if (rc) return -630 - rc;
    for (int i = 0; i < n_out; i++){
        char finalpath[600];
        if (g_dump_dir){
            // 标定模式：忽略 spec 里的路径，统一写 <dir>/o<slot>.bin
            snprintf(finalpath, sizeof finalpath, "%s/o%d.bin", g_dump_dir, oidx[i] >= 0 ? oidx[i] : i);
        } else {
            snprintf(finalpath, sizeof finalpath, "%s", opath[i]);
        }
        FILE* f = fopen(finalpath, "wb");
        if (!f) return -604;
        fwrite(outs[i], 1, osz[i], f);
        fclose(f);
        free(outs[i]);
    }
    return 0;
}

int main(void){
    if (!load_adapter()){ reply("{\"ok\":false,\"err\":\"adapter load failed\"}"); return 1; }
    reply("{\"ok\":true,\"devices\":[\"mtk-gpu\",\"mtk-dsp\",\"mtk-mdla\"],\"ready\":true}");
    char line[8192];
    while (fgets(line, sizeof line, stdin)){
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        size_t L = strlen(p);
        while (L && (p[L-1] == '\n' || p[L-1] == '\r')) p[--L] = 0;
        if (!L) continue;
        if (!strncmp(p, "{\"cmd\":\"quit\"", 12)) break;
        if (!strncmp(p, "{\"cmd\":\"build\",\"path\":\"", 23)){
            char* q = p + 23; char path[512]; int k = 0;
            while (*q && *q != '"' && k < 511) path[k++] = *q++;
            path[k] = 0;
            if (g_nslots >= MAXSLOT){ reply("{\"ok\":false,\"err\":\"slots full\"}"); continue; }
            g_slot = g_nslots++;
            int rc = build_from_file(path);
            if (rc < 0){ printf("{\"ok\":false,\"err\":\"build\",\"rc\":%d}\n", rc); fflush(stdout); }
            else { printf("{\"ok\":true,\"ops\":%d,\"slot\":%d}\n", rc, g_slot); fflush(stdout); }
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"create\"", 14)){
            void* m = 0;
            int rc = M_create(&m);
            g_models[g_slot] = m;
            printf("{\"ok\":%s,\"rc\":%d}\n", rc == 0 ? "true" : "false", rc);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"compile\"", 15)){
            int rc = compile_model();
            printf("{\"ok\":%s,\"rc\":%d}\n", rc == 0 ? "true" : "false", rc);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"run\",\"spec\":\"", 21)){
            char* q = p + 21; char spec[16384]; int k = 0;
            while (*q && *q != '"' && k < 16383) spec[k++] = *q++;
            spec[k] = 0;
            int rc = run_spec(spec);
            printf("{\"ok\":%s,\"rc\":%d}\n", rc == 0 ? "true" : "false", rc);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"dbg\",\"path\":\"", 20)){
            // 开启调试报告：设置 report path 并改用 createForDebug
            char* q = p + 20; char path[512]; int k = 0;
            while (*q && *q != '"' && k < 511) path[k++] = *q++;
            path[k] = 0;
            int rc = -1;
            if (M_dbgpath && g_models[g_slot]) rc = M_dbgpath(g_models[g_slot], path);
            g_usedbg = (rc == 0);
            printf("{\"ok\":%s,\"rc\":%d,\"dbg\":%d}\n", rc == 0 ? "true" : "false", rc, g_usedbg);
            fflush(stdout);
            continue;
        }        if (!strncmp(p, "{\"cmd\":\"dump\",\"dir\":\"", 21)){
            // 标定模式：设置把所有输出写到指定目录
            char* q = p + 21; char path[512]; int k = 0;
            while (*q && *q != '"' && k < 511) path[k++] = *q++;
            path[k] = 0;
            if (!path[0]){ g_dump_dir = 0; reply("{\"ok\":true,\"dump\":\"off\"}"); continue; }
            static char dbuf[512];
            snprintf(dbuf, sizeof dbuf, "%s", path);
            g_dump_dir = dbuf;
            printf("{\"ok\":true,\"dump\":\"%s\",\"n_out\":%d}\n", dbuf, g_n_out[g_slot]);
            fflush(stdout);
            continue;
        }        if (!strncmp(p, "{\"cmd\":\"export\",\"path\":\"", 24)){
            char* q = p + 24; char path[512]; int k = 0;
            while (*q && *q != '"' && k < 511) path[k++] = *q++;
            path[k] = 0;
            int rc = export_blob(path);
            printf("{\"ok\":%s,\"rc\":%d,\"bytes\":%ld}\n", rc == 0 ? "true" : "false", rc, g_blob_size);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"pop\"", 12)){
            // 释放最后一个 slot，让下一个 rpush 复用（测试 IOVA 能否回收）
            if (g_nslots <= 0){ reply("{\"ok\":false,\"err\":\"empty\"}"); continue; }
            int s = g_nslots - 1;
            int rf = -1, cf = -1;
            if (C_free && g_comps[s]){ cf = C_free(g_comps[s]); g_comps[s] = 0; }
            if (M_free && g_models[s]){ rf = M_free(g_models[s]); g_models[s] = 0; }
            if (g_rst_buf[s]){ free(g_rst_buf[s]); g_rst_buf[s] = 0; }
            g_nslots_restored[s] = 0;
            g_n_in[s] = g_n_out[s] = 0;
            g_nslots--;
            if (g_slot >= g_nslots) g_slot = g_nslots > 0 ? g_nslots - 1 : 0;
            printf("{\"ok\":true,\"freed_slot\":%d,\"model_free\":%d,\"comp_free\":%d,\"slots\":%d}\n",
                   s, rf, cf, g_nslots);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"bload\",\"list\":\"", 23)){
            // 一次性把所有 blob 读进常驻内存池（之后 map 不再碰磁盘）
            char* q = p + 23; char path[512]; int k = 0;
            while (*q && *q != '"' && k < 511) path[k++] = *q++;
            path[k] = 0;
            FILE* lf = fopen(path, "r");
            if (!lf){ printf("{\"ok\":false,\"err\":\"list open\"}\n"); fflush(stdout); continue; }
            char line[512];
            while (fgets(line, sizeof line, lf) && g_nblob < MAXBLOB){
                char* e = line; while (*e && *e != '\n' && *e != '\r') e++; *e = 0;
                if (!line[0]) continue;
                long sz = file_size(line);
                if (sz <= 0) continue;
                void* b = malloc((size_t)sz);
                if (!b) continue;
                FILE* bf = fopen(line, "rb");
                if (!bf || fread(b, 1, (size_t)sz, bf) != (size_t)sz){
                    if (bf) fclose(bf); free(b); continue;
                }
                fclose(bf);
                g_blob_ptr[g_nblob] = b;
                g_blob_len[g_nblob] = sz;
                g_nblob++;
            }
            fclose(lf);
            printf("{\"ok\":true,\"n\":%d}\n", g_nblob);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"map\",\"idx\":", 19)){
            // 从常驻内存池映射第 idx 个块；固定使用最后一个 slot
            int idx = (int)strtol(p + 19, 0, 10);
            if (idx < 0 || idx >= g_nblob){
                printf("{\"ok\":false,\"err\":\"bad idx\",\"n\":%d}\n", g_nblob); fflush(stdout); continue;
            }
            int s = MAXSLOT - 1;
            if (g_mapped){ release_slot(s); g_mapped = 0; }
            g_slot = s;
            int rc = restore_from_buf(g_blob_ptr[idx], g_blob_len[idx], idx);
            if (rc != 0){
                printf("{\"ok\":false,\"err\":\"map\",\"rc\":%d}\n", rc); fflush(stdout); continue;
            }
            g_mapped = 1;
            printf("{\"ok\":true,\"idx\":%d,\"nin\":%d,\"nout\":%d}\n",
                   idx, g_n_in[s], g_n_out[s]);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"unmap\"", 14)){
            // 用完立即甩掉，把 IOVA 还给系统
            if (!g_mapped){ reply("{\"ok\":true,\"already\":true}"); continue; }
            release_slot(MAXSLOT - 1);
            g_mapped = 0;
            reply("{\"ok\":true}");
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"rpush\",\"path\":\"", 23)){
            // 把 blob 恢复进一个新 slot：跳过 build 与 compile，只需要一次顺序读
            char* q = p + 23; char path[512]; int k = 0;
            while (*q && *q != '"' && k < 511) path[k++] = *q++;
            path[k] = 0;
            if (g_nslots >= MAXSLOT){ reply("{\"ok\":false,\"err\":\"slots full\"}"); continue; }
            int saved = g_slot;
            g_slot = g_nslots;
            int rc = restore_blob(path);
            if (rc != 0){
                g_slot = saved;
                printf("{\"ok\":false,\"err\":\"restore\",\"rc\":%d}\n", rc); fflush(stdout);
                continue;
            }
            g_nslots++;
            printf("{\"ok\":true,\"slot\":%d,\"nin\":%d,\"nout\":%d}\n",
                   g_slot, g_n_in[g_slot], g_n_out[g_slot]);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"restore\",\"path\":\"", 25)){
            char* q = p + 25; char path[512]; int k = 0;
            while (*q && *q != '"' && k < 511) path[k++] = *q++;
            path[k] = 0;
            int rc = restore_blob(path);
            printf("{\"ok\":%s,\"rc\":%d}\n", rc == 0 ? "true" : "false", rc);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"sup\"", 12)){
            // 查询当前 model 在 mtk-mdla 上哪些算子不支持
            if (!M_supd || !h_dev_mdla){ reply("{\"ok\":false,\"err\":\"no api\"}"); continue; }
            int n = g_n_op_count[g_slot];
            if (n <= 0){ reply("{\"ok\":false,\"err\":\"no ops\"}"); continue; }
            bool* sup = (bool*)calloc((size_t)n, sizeof(bool));
            if (!sup){ reply("{\"ok\":false,\"err\":\"oom\"}"); continue; }
            const void* devs[1] = { h_dev_mdla };
            int rc = M_supd(g_models[g_slot], devs, 1, sup);
            if (rc){ free(sup); printf("{\"ok\":false,\"rc\":%d}\n", rc); fflush(stdout); continue; }
            int bad = 0;
            for (int i = 0; i < n; i++) if (!sup[i]) bad++;
            printf("{\"ok\":true,\"total\":%d,\"unsupported\":%d,\"first\":[", n, bad);
            int shown = 0;
            for (int i = 0; i < n && shown < 20; i++){
                if (sup[i]) continue;
                printf("%s{\"idx\":%d,\"code\":%d}", shown ? "," : "", i, g_op_code[g_slot][i]);
                shown++;
            }
            printf("]}\n");
            fflush(stdout);
            free(sup);
            continue;
        }        if (!strncmp(p, "{\"cmd\":\"ops\"", 12)){
            if (!M_sup){ reply("{\"ok\":false,\"err\":\"no api\"}"); continue; }
            void* m = 0;
            M_create(&m);
            uint32_t cap = 1024;
            uint32_t* list = (uint32_t*)malloc(sizeof(uint32_t)*cap);
            uint32_t cnt = cap;
            int rc = M_sup(m, &cnt, list);
            if (rc == 0){
                printf("{\"ok\":true,\"count\":%u,\"ops\":[", cnt);
                for (uint32_t i = 0; i < cnt && i < cap; i++)
                    printf("%s%u", i ? "," : "", list[i]);
                printf("]}\n");
            } else printf("{\"ok\":false,\"rc\":%d,\"cnt\":%u}\n", rc, cnt);
            free(list);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"use\",\"slot\":", 20)){
            int s = atoi(p + 20);
            if (s < 0 || s >= g_nslots){ reply("{\"ok\":false,\"err\":\"bad slot\"}"); continue; }
            g_slot = s;
            printf("{\"ok\":true,\"slot\":%d}\n", s); fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"cachedir\",\"path\":\"", 26)){
            char* q = p + 26; char path[512]; int k = 0;
            while (*q && *q != '"' && k < 511) path[k++] = *q++;
            path[k] = 0;
            strncpy(g_cache_dir, path, sizeof g_cache_dir - 1);
            printf("{\"ok\":true,\"dir\":\"%s\"}\n", g_cache_dir); fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"cachekey\",\"v\":\"", 23)){
            char* q = p + 23; char v[512]; int k = 0;
            while (*q && *q != '"' && k < 511) v[k++] = *q++;
            v[k] = 0;
            strncpy(g_cache_key, v, sizeof g_cache_key - 1);
            printf("{\"ok\":true}\n"); fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"chain\",\"spec\":\"", 23)){
            char* q = p + 23; char spec[8192]; int k = 0;
            while (*q && *q != '"' && k < 8191) spec[k++] = *q++;
            spec[k] = 0;
            int rc = chain_run(spec);
            printf("{\"ok\":%s,\"rc\":%d,\"steps\":%d}\n", rc > 0 ? "true" : "false", rc, rc > 0 ? rc : 0);
            fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"nodev\"", 14)){
            char* vq = strstr(p, "\"v\":");
            g_nodev = vq ? (atoi(vq + 4) ? 1 : 0) : 1;
            printf("{\"ok\":true,\"nodev\":%d}\n", g_nodev); fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"relax\"", 14)){
            char* vq = strstr(p, "\"v\":");
            g_relax = vq ? atoi(vq + 4) : 1;
            g_relax = g_relax ? 1 : 0;
            printf("{\"ok\":true,\"relax\":%d}\n", g_relax); fflush(stdout);
            continue;
        }
        if (!strncmp(p, "{\"cmd\":\"poolclear\"", 18)){
            pool_clear(); reply("{\"ok\":true}"); continue;
        }
        if (!strncmp(p, "{\"cmd\":\"ping\"", 13)){ reply("{\"ok\":true,\"pong\":1}"); continue; }
        reply("{\"ok\":false,\"err\":\"unknown cmd\"}");
    }
    return 0;
}
