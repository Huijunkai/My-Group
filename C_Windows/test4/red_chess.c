#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<sys/ipc.h>
#include<semaphore.h>
#include<fcntl.h>
#include<unistd.h>
/* 下象棋示例红棋red_chess.c */

int main(int argc, char *argv[]){
    int i = 0;
    sem_t *hei = sem_open("chess_black_sem", O_CREAT, 0666, 1);
    sem_t *hong = sem_open("chess_red_sem", O_CREAT, 0666, 0);
    for ( i = 0; i < 10; i++){
        sem_wait(hei);
        if (i != 9)
        {
            printf("Red chess had moved, black chess go!\n");
        }else{
            printf("Red chess win !!!\n");
        }
        fflush(stdout);
        sem_post(hong);
    }
    sleep(10);
    sem_close(hei);
    sem_close(hong);
    sem_unlink("chess_red_sem");
    sem_unlink("chess_black_sem");
    exit(0);
}