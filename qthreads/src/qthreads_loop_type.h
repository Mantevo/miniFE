#ifndef QTHREADS_LOOP_TYPES_H
#define QTHREADS_LOOP_TYPES_H

#ifdef USING_QLOOP_QT_LOOP
#define QLOOP(start,stop,func,args) do { qt_loop(start,stop,func,args); } while (0)
#elif USING_QLOOP_QT_LOOP_BALANCE
#define QLOOP(start,stop,func,args) do { qt_loop_balance(start,stop,func,args); } while (0)
#elif USING_QLOOP_QT_LOOP_QUEUE_CHUNK
#define QLOOP(start,stop,func,args) \
    do { \
        qqloop_handle_t *l = NULL; \
        l = qt_loop_queue_create(CHUNK, start, stop, 1, func, args); \
        qt_loop_queue_run(l); \
    } while (0)
#elif USING_QLOOP_QT_LOOP_QUEUE_GUIDED
#define QLOOP(start,stop,func,args) \
    do { \
        qqloop_handle_t *l = NULL; \
        l = qt_loop_queue_create(GUIDED, start, stop, 1, func, args); \
        qt_loop_queue_run(l); \
    } while (0)
#elif USING_QLOOP_QT_LOOP_QUEUE_FACTORED
#define QLOOP(start,stop,func,args) \
    do { \
        qqloop_handle_t *l = NULL; \
        l = qt_loop_queue_create(FACTORED, start, stop, 1, func, args); \
        qt_loop_queue_run(l); \
    } while (0)
#elif USING_QLOOP_QT_LOOP_QUEUE_TIMED
#define QLOOP(start,stop,func,args) \
    do { \
        qqloop_handle_t *l = NULL; \
        l = qt_loop_queue_create(TIMED, start, stop, 1, func, args); \
        qt_loop_queue_run(l); \
    } while (0)
#else
#error "unimplemented loop choice"
#endif

#endif
