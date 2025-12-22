#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<sys/ipc.h>
#include<semaphore.h>
#include<fcntl.h>
#include<sys/stat.h>
#include<unistd.h>

int main(int argc, char *argv[]){
    char message = 'x';
    int i = 0;
    if (argc >1){
        message = argv[1][0];
    }
    sem_t *mutex=sem_open("mysem", O_CREAT,0666,1);
    for(i = 0; i <10; i++){
        sem_wait(mutex);
        printf("%c",message);
        fflush(stdout);
        sleep(rand()%3);
        printf("%c",message);
        fflush(stdout);
        sem_post(mutex);
        sleep(rand()%2);
    }
    sleep(10);
    sem_close(mutex);
    sem_unlink("mysem");
    exit(0);
}