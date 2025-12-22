、#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<sys/ipc.h>
#include<semaphore.h>
#include<fcntl.h>
#include<unistd.h>

/* 下象棋示例黑棋black_chess.c */
int main(int argc, char *argv[]){
    int i = 0;
    sem_t *hei = sem_open("chess_black_sem", O_CREAT,0666,1);
    sem_t *hong = sem_open("chess_red_sem", O_CREAT, 0666, 0);
    for ( i = 0; i < 10; i++)
    {
        sem_wait(hong);
        if (i != 9)
        {
            printf("Black chess had moved, red chess got!\n");
            /* code */
        }else{
            printf("Black chess lost!!!\n");
        }
        fflush(stdout);
        sem_post(hei); 
    }
    sleep(10);
    sem_close(hei);
    sem_close(hong);
    sem_unlink("chess_red_sem");
    sem_unlink("chess_black_sem");
    exit(0);
}