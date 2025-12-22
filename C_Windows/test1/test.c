#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>

int main(){
	pid_t son_pid, daughter_pid;
	int i,count =1;
	for(i=1;i<3;i++){
		son_pid = fork();
		if(son_pid == 0){
			count++;
			printf("I am son,count=%d\n",count);
		}else {
			daughter_pid = fork();
			if(daughter_pid == 0){
				count++;
				printf("I am daughter, count = %d\n",count);
			}else{
				count++;
				printf("I am father, count = %d\n",count);
				waitpid(son_pid,NULL,0);
				waitpid(daughter_pid, NULL, 0);
			}
		}
	}
	return 0;
}