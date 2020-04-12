#ifndef __DAG_TASK_QUEUE_H__
#define __DAG_TASK_QUEUE_H__

struct DAG_task_queue
{
    int max_task_id;    // Max task id + 1
    int num_task;       // Number of actual tasks
    int task_head;      // Head index of currently avail tasks in the queue
    int task_tail;      // Tail index of currently avail tasks in the queue
    int *DAG_src_ptr;   // Size max_task_id+1, DAG CSR matrix row_ptr array
    int *DAG_dst_idx;   // Size unknown, DAG CSR matrix col_idx array
    int *indeg;         // Size max_task_id, indegree of DAG vertexes 
    int *curr_indeg;    // Size max_task_id, indegree of DAG vertexes in running
    int *task_queue;    // Size max_task_id, task queue
};
typedef struct DAG_task_queue  DAG_task_queue_s;
typedef struct DAG_task_queue* DAG_task_queue_t;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a DAG_task_queue structure with a DAG stored in CSR matrix. 
// DAG(i, j) is nonzero means that task j relies on task i. If DAG(i, i) is 
// nonzero, task i will be skipped. 
// Input parameters:
//   max_task_id : Max task id + 1, Tasks are indexed from 0 to max_task_id-1
//   num_dep     : Number of dependencies (nonzeros in DAG matrix)
//   DAG_src_ptr : Size max_task_id+1, CSR matrix row_ptr array
//   DAG_dst_idx : Size num_dep, CSR matrix col_idx array
// Output parameter:
//   *tq_ : Pointer to an initialized DAG_task_queue structure
void DAG_task_queue_init(
    const int max_task_id, const int num_dep, const int *DAG_src_ptr, 
    const int *DAG_dst_idx, DAG_task_queue_t *tq_
);

// Destroy a DAG_task_queue structure.
// Input parameter:
//   tq : A DAG_task_queue structure to be destroyed
void DAG_task_queue_free(DAG_task_queue_t tq);

// Get a new task from a DAG_task_queue structure and update its task queue.  
// This function can be called by multiple threads at the same time.
// Input parameter:
//   tq : Target DAG_task_queue structure
// Output parameters:
//   tq       : Target DAG_task_queue structure with updated task queue info
//   <return> : Index of the new task. -1 means all tasks are finished.
int  DAG_task_queue_get_task(DAG_task_queue_t tq);

// Finish a task and push new available tasks to a DAG_task_queue task queue.
// This function can be called by multiple threads at the same time.
// Input parameters:
//   tq      : Target DAG_task_queue structure
//   task_id : Index of the finished task
// Output parameter:
//   tq : Target DAG_task_queue structure with updated task queue info
void DAG_task_queue_finish_task(DAG_task_queue_t tq, const int task_id);

// Reset the task queue in a DAG_task_queue structure. 
// Input parameter:
//   tq : Target DAG_task_queue structure
// Output parameters:
//   tq : Target DAG_task_queue structure with updated task queue info
void DAG_task_queue_reset(DAG_task_queue_t tq);

#ifdef __cplusplus
}
#endif

#endif  // End of "#ifndef __DAG_TASK_QUEUE_H__"

