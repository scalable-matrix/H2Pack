#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include "DAG_task_queue.h"

// Initialize a DAG_task_queue structure with a DAG stored in CSR format. 
// DAG(i, j) is nonzero means that task j relies on task i. If DAG(i, i) is 
// nonzero, task i will be skipped. 
void DAG_task_queue_init(
    const int max_task_id, const int num_dep, const int *DAG_src_ptr, 
    const int *DAG_dst_idx, DAG_task_queue_p *tq_
)
{
    DAG_task_queue_p tq = (DAG_task_queue_p) malloc(sizeof(DAG_task_queue_s));
    assert(tq != NULL);
    
    // Allocate arrays in DAG_task_queue
    tq->DAG_src_ptr = (int*) malloc(sizeof(int) * (max_task_id + 1));
    tq->DAG_dst_idx = (int*) malloc(sizeof(int) * num_dep);
    tq->indeg       = (int*) malloc(sizeof(int) * max_task_id);
    tq->curr_indeg  = (int*) malloc(sizeof(int) * max_task_id);
    tq->task_queue  = (int*) malloc(sizeof(int) * max_task_id);
    assert(tq->DAG_src_ptr != NULL);
    assert(tq->DAG_dst_idx != NULL);
    assert(tq->indeg       != NULL);
    assert(tq->curr_indeg  != NULL);
    assert(tq->task_queue  != NULL);
    
    // Copy DAG CSR matrix, count DAG vertex indegree and number of actual tasks
    if (num_dep != DAG_src_ptr[max_task_id])
    {
        fprintf(stderr, "ERROR: num_dep != DAG_src_ptr[max_task_id] \n");
        return;
    }
    tq->max_task_id = max_task_id;
    tq->num_task    = max_task_id;
    memcpy(tq->DAG_src_ptr, DAG_src_ptr, sizeof(int) * (max_task_id + 1));
    memcpy(tq->DAG_dst_idx, DAG_dst_idx, sizeof(int) * num_dep);
    memset(tq->indeg, 0, sizeof(int) * max_task_id);
    for (int i = 0; i < max_task_id; i++)
    {
        for (int j = tq->DAG_src_ptr[i]; j < tq->DAG_src_ptr[i + 1]; j++)
        {
            int dst = tq->DAG_dst_idx[j];
            if (dst == i) tq->num_task--;  // Task i relies on task i
            tq->indeg[dst]++;
        }
    }
    
    DAG_task_queue_reset(tq);
    
    *tq_ = tq;
}

// Destroy a DAG_task_queue structure.
void DAG_task_queue_free(DAG_task_queue_p tq)
{
    if (tq == NULL) return;
    free(tq->DAG_src_ptr);
    free(tq->DAG_dst_idx);
    free(tq->indeg);
    free(tq->curr_indeg);
    free(tq->task_queue);
    free(tq);
}

// Get a new task from a DAG_task_queue structure and update its task queue.  
// This function can be called by multiple threads at the same time.
int  DAG_task_queue_get_task(DAG_task_queue_p tq)
{
    if (tq == NULL) return -1;
    
    // Get current task queue head index and increment it
    // If all tasks are finished, return directly
    int task_head = __atomic_fetch_add(&tq->task_head, 1, __ATOMIC_SEQ_CST);
    if (task_head >= tq->num_task) return -1;
    
    // Atomic load the task id, task_id = -1 means the task_head-th task is not 
    // available yet, otherwise we have a valid task_id and return
    int task_id = __atomic_load_n(&tq->task_queue[task_head], __ATOMIC_SEQ_CST);
    while (task_id == -1)
    {
        //usleep(10);
        task_id = __atomic_load_n(&tq->task_queue[task_head], __ATOMIC_SEQ_CST);
    }
    //if (task_id == -1) printf("[Warning] task_head = %d, task_id = -1\n", task_head);
    return task_id;
}

// Finish a task and push new available tasks to a DAG_task_queue task queue.
// This function can be called by multiple threads at the same time.
void DAG_task_queue_finish_task(DAG_task_queue_p tq, const int task_id)
{
    if (tq == NULL) return;
    for (int j = tq->DAG_src_ptr[task_id]; j < tq->DAG_src_ptr[task_id + 1]; j++)
    {
        // For a destination vertex, subtract its current indegree count by 1
        // and get its new indegree count to see if it is available now
        int dst = tq->DAG_dst_idx[j];
        int dst_indeg = __atomic_sub_fetch(tq->curr_indeg + dst, 1, __ATOMIC_SEQ_CST);
        
        // If the destination vertex is now available, push it to task queue
        if (dst_indeg == 0)
        {
            int task_tail = __atomic_fetch_add(&tq->task_tail, 1, __ATOMIC_SEQ_CST);
            __atomic_store_n(&tq->task_queue[task_tail], dst, __ATOMIC_SEQ_CST);
        }
        //if (dst_indeg < 0) printf("Warning: from task %d, set %d indeg = %d\n", task_id, dst, dst_indeg);
    }
}

// Reset the task queue in a DAG_task_queue structure. 
void DAG_task_queue_reset(DAG_task_queue_p tq)
{
    if (tq == NULL) return;
    tq->task_head = 0;
    tq->task_tail = 0;
    for (int i = 0; i < tq->max_task_id; i++)
    {
        tq->curr_indeg[i] = tq->indeg[i];
        tq->task_queue[i] = -1;  // Mark all the tasks in the queue as unavailable
        if (tq->indeg[i] == 0)
        {
            tq->task_queue[tq->task_tail] = i;
            tq->task_tail++;
        }
    }
}
