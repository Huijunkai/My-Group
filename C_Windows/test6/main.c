#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

// 作业结构体（已分配内存的作业）
struct Jobs {
    int jobID;
    int addr_start;
    int addr_end;
    struct Jobs *next;
};

// 空闲分区结构体（双向链表）
struct Unoccupied_block {
    struct Unoccupied_block *previous;
    int addr_start;
    int addr_end;
    struct Unoccupied_block *next;
};

// 阻塞队列结构体（未分配到内存的作业）
struct Blocking_queue {
    int jobID;
    int space;
    struct Blocking_queue *next;
};

// 全局链表头指针（统一初始化）
struct Jobs *jobhead;
struct Blocking_queue *bqhead;
struct Unoccupied_block *ubhead;

// 判断阻塞队列是否为空
bool empty() {
    return bqhead->next == NULL;
}

// 排序1：按空闲分区大小从小到大排序（最佳适应算法用）
void sort_by_size() {
    struct Unoccupied_block *t = ubhead->next;
    int length = 0;
    // 计算空闲分区数量
    while (t) {
        length++;
        t = t->next;
    }
    if (length <= 1) return; // 只有1个分区，无需排序

    // 冒泡排序（交换节点数据，避免指针操作复杂）
    for (int i = 0; i < length - 1; i++) {
        t = ubhead->next;
        for (int j = 0; j < length - i - 1; j++) {
            struct Unoccupied_block *a = t;
            struct Unoccupied_block *b = t->next;
            // 计算分区大小
            int size_a = a->addr_end - a->addr_start;
            int size_b = b->addr_end - b->addr_start;
            // 按大小升序排序
            if (size_a > size_b) {
                // 交换a和b的数据（不交换指针，简化逻辑）
                int temp_start = a->addr_start;
                int temp_end = a->addr_end;
                a->addr_start = b->addr_start;
                a->addr_end = b->addr_end;
                b->addr_start = temp_start;
                b->addr_end = temp_end;
            }
            t = t->next;
        }
    }
}

// 排序2：按空闲分区起始地址从小到大排序（首次适应算法用）
void sort_by_addr() {
    struct Unoccupied_block *t = ubhead->next;
    int length = 0;
    while (t) {
        length++;
        t = t->next;
    }
    if (length <= 1) return; // 只有1个分区，无需排序

    // 冒泡排序（按起始地址升序）
    for (int i = 0; i < length - 1; i++) {
        t = ubhead->next;
        for (int j = 0; j < length - i - 1; j++) {
            struct Unoccupied_block *a = t;
            struct Unoccupied_block *b = t->next;
            if (a->addr_start > b->addr_start) {
                // 交换数据
                int temp_start = a->addr_start;
                int temp_end = a->addr_end;
                a->addr_start = b->addr_start;
                a->addr_end = b->addr_end;
                b->addr_start = temp_start;
                b->addr_end = temp_end;
            }
            t = t->next;
        }
    }
}

// 输出当前空闲分区链
void Output() {
    struct Unoccupied_block *t = ubhead->next;
    int count = 0;
    printf("------------------------------------\n");
    printf("当前的空闲分区链为：\n");
    while (t) {
        count++;
        // 输出格式：序号 起始地址 结束地址（分区大小=结束-起始）
        printf("%d %d %d\n", count, t->addr_start, t->addr_end);
        t = t->next;
    }
    printf("------------------------------------\n\n");
}

// 向已分配作业链表插入新作业
void InsertJob(struct Jobs newJob) {
    struct Jobs *t1 = jobhead;
    // 找到链表尾部
    while (t1->next) {
        t1 = t1->next;
    }
    // 分配内存（避免局部变量销毁问题）
    struct Jobs *t = (struct Jobs *)malloc(sizeof(struct Jobs));
    t->jobID = newJob.jobID;
    t->addr_start = newJob.addr_start;
    t->addr_end = newJob.addr_end;
    t->next = NULL;
    t1->next = t;
}

// 向阻塞队列插入作业
void InsertBQ(struct Blocking_queue newJob) {
    struct Blocking_queue *t = bqhead;
    while (t->next) {
        t = t->next;
    }
    struct Blocking_queue *t2 = (struct Blocking_queue *)malloc(sizeof(struct Blocking_queue));
    t2->jobID = newJob.jobID;
    t2->space = newJob.space;
    t2->next = NULL;
    t->next = t2;
}

// 从作业链表删除指定ID的作业（修复段错误：添加空指针检查）
bool DeleteJob(int id, int *start, int *end) {
    struct Jobs *t = jobhead;
    // 遍历查找作业
    while (t->next) {
        if (t->next->jobID == id) {
            break;
        }
        t = t->next;
    }
    // 关键：检查是否找到作业（避免t->next为NULL）
    if (t->next == NULL) {
        printf("错误：未找到ID为%d的作业，回收失败！\n", id);
        return false;
    }
    // 保存作业地址信息
    *start = t->next->addr_start;
    *end = t->next->addr_end;
    // 释放节点内存（避免内存泄漏）
    struct Jobs *temp = t->next;
    t->next = t->next->next;
    free(temp);

    printf("回收成功\n");
    printf("作业信息为：ID: %d Space: %d\n", id, *end - *start);
    return true;
}

// 从阻塞队列删除指定ID的作业（返回删除后的下一个节点）
struct Blocking_queue *DeleteBQ(int id) {
    struct Blocking_queue *t = bqhead;
    while (t->next) {
        if (t->next->jobID == id) {
            break;
        }
        t = t->next;
    }
    // 检查是否找到（避免空指针）
    if (t->next == NULL) {
        return NULL;
    }
    struct Blocking_queue *temp = t->next;
    t->next = t->next->next;
    free(temp); // 释放内存
    return t->next;
}

// 内存分配核心逻辑（blockFlag：是否为阻塞队列的作业）
bool Arrange(int newJobID, int newJobSpace, bool blockFlag) {
    struct Unoccupied_block *head1 = ubhead->next;
    bool flag = false;

    while (head1) {
        int block_size = head1->addr_end - head1->addr_start;
        // 情况1：空闲分区大于作业大小（分割分区）
        if (block_size > newJobSpace) {
            printf("分配成功\n");
            printf("作业信息为：ID: %d Space: %d\n", newJobID, newJobSpace);
            // 构造新作业
            struct Jobs newjob;
            newjob.addr_start = head1->addr_start;
            newjob.addr_end = newjob.addr_start + newJobSpace;
            newjob.jobID = newJobID;
            newjob.next = NULL;
            InsertJob(newjob);
            // 更新空闲分区起始地址
            head1->addr_start += newJobSpace;
            flag = true;
            break;
        }
        // 情况2：空闲分区等于作业大小（删除空闲分区）
        else if (block_size == newJobSpace) {
            printf("分配成功\n");
            printf("作业信息为：ID: %d Space: %d\n", newJobID, newJobSpace);
            struct Jobs newjob;
            newjob.addr_start = head1->addr_start;
            newjob.addr_end = head1->addr_end;
            newjob.jobID = newJobID;
            newjob.next = NULL;
            InsertJob(newjob);
            // 删除当前空闲分区（双向链表操作）
            head1->previous->next = head1->next;
            if (head1->next != NULL) {
                head1->next->previous = head1->previous;
            }
            free(head1); // 释放空闲分区节点
            flag = true;
            break;
        }
        // 情况3：分区太小，继续查找下一个
        else {
            head1 = head1->next;
        }
    }

    // 分配失败：加入阻塞队列（非阻塞队列作业才加入）
    if (!flag) {
        printf("分配失败\n");
        printf("作业信息为：ID: %d Space: %d\n", newJobID, newJobSpace);
        if (!blockFlag) {
            struct Blocking_queue newJob;
            newJob.jobID = newJobID;
            newJob.space = newJobSpace;
            newJob.next = NULL;
            InsertBQ(newJob);
        }
        return false;
    }
    return true;
}

// 内存回收核心逻辑（合并邻接碎片）
void Free(int newJobID) {
    int jobAddrStart = -1, jobAddrEnd = -1;
    // 先删除作业，若删除失败直接返回
    if (!DeleteJob(newJobID, &jobAddrStart, &jobAddrEnd)) {
        Output();
        return;
    }

    // 回收前先按地址排序（确保邻接分区在附近，方便合并）
    sort_by_addr();

    struct Unoccupied_block *indexInsert = ubhead;
    struct Unoccupied_block *newItem = (struct Unoccupied_block *)malloc(sizeof(struct Unoccupied_block));
    newItem->addr_start = jobAddrStart;
    newItem->addr_end = jobAddrEnd;
    newItem->previous = NULL;
    newItem->next = NULL;

    // 寻找插入位置（按地址顺序）
    while (indexInsert->next != NULL && indexInsert->next->addr_start < jobAddrEnd) {
        indexInsert = indexInsert->next;
    }

    // 插入新回收的分区（双向链表操作）
    newItem->next = indexInsert->next;
    newItem->previous = indexInsert;
    if (indexInsert->next != NULL) {
        indexInsert->next->previous = newItem;
    }
    indexInsert->next = newItem;

    // 合并碎片（检查前后是否邻接）
    // 1. 与后一个分区合并
    if (newItem->next != NULL && newItem->addr_end == newItem->next->addr_start) {
        struct Unoccupied_block *temp = newItem->next;
        newItem->addr_end = temp->addr_end;
        newItem->next = temp->next;
        if (temp->next != NULL) {
            temp->next->previous = newItem;
        }
        free(temp);
    }
    // 2. 与前一个分区合并
    if (newItem->previous != ubhead && newItem->previous->addr_end == newItem->addr_start) {
        struct Unoccupied_block *temp = newItem;
        newItem = newItem->previous;
        newItem->addr_end = temp->addr_end;
        newItem->next = temp->next;
        if (temp->next != NULL) {
            temp->next->previous = newItem;
        }
        free(temp);
    }
}

// 算法入口（altype：false=首次适应，true=最佳适应）
void memory_allocate(bool altype) {
    FILE *fp;
    printf("请输入文件名：\n");
    char filename[20];
    scanf("%s", filename);
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("打开文件错误\n");
        return;
    }

    // 初始化三大链表（头结点，不存储实际数据）
    jobhead = (struct Jobs *)malloc(sizeof(struct Jobs));
    jobhead->next = NULL;
    bqhead = (struct Blocking_queue *)malloc(sizeof(struct Blocking_queue));
    bqhead->next = NULL;
    ubhead = (struct Unoccupied_block *)malloc(sizeof(struct Unoccupied_block));
    ubhead->previous = NULL;
    ubhead->next = NULL;
    ubhead->addr_start = -1;
    ubhead->addr_end = -1;

    // 初始化空闲分区（总内存640KB，地址0~640）
    struct Unoccupied_block *first_block = (struct Unoccupied_block *)malloc(sizeof(struct Unoccupied_block));
    first_block->addr_start = 0;
    first_block->addr_end = 640;
    first_block->previous = ubhead;
    first_block->next = NULL;
    ubhead->next = first_block;

    // 读取文件中的作业（格式：作业ID 操作类型（1=分配/0=回收） 空间大小）
    int id, type, space;
    while (fscanf(fp, "%d %d %d", &id, &type, &space) == 3) { // 修复feof读取bug
        if (type == 1) { // 分配作业
            // 首次适应：分配前按地址排序（低地址优先）
            if (!altype) {
                sort_by_addr();
            }
            // 最佳适应：分配前按大小排序（最小分区优先）
            else {
                sort_by_size();
            }
            Arrange(id, space, false);
            Output();
        } else if (type == 0) { // 回收作业
            Free(id);
            // 回收后按算法类型重新排序
            if (altype) {
                sort_by_size();
            } else {
                sort_by_addr();
            }
            // 处理阻塞队列（循环尝试分配，直到失败）
            if (!empty()) {
                struct Blocking_queue *t = bqhead->next;
                while (t) {
                    int currentID = t->jobID;
                    int currentSpace = t->space;
                    printf("处理阻塞队列中的作业%d\n", currentID);
                    // 分配前重新排序（确保算法逻辑正确）
                    if (altype) sort_by_size();
                    else sort_by_addr();
                    // 分配成功则删除阻塞队列中的作业，重新遍历
                    if (Arrange(currentID, currentSpace, true)) {
                        t = DeleteBQ(currentID);
                        Output();
                    } else {
                        t = t->next;
                    }
                }
            }
        }
    }

    // 关闭文件+释放内存（简化版，完整版可遍历所有节点释放）
    fclose(fp);
    free(jobhead);
    free(bqhead);
    free(ubhead);
}

int main(void) {
    printf("*******************************************************\n\n");
    printf("首次适应算法：\n\n");
    memory_allocate(false); // 首次适应：altype=false

    printf("*******************************************************\n\n");
    printf("最佳适应算法：\n\n");
    memory_allocate(true);  // 最佳适应：altype=true

    return 0;
}
