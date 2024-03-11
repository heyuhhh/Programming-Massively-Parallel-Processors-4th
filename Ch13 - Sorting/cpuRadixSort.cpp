#include<iostream>
#include<algorithm>  //max_element
#include<cstring>    //memset, memcpy
#include<time.h>
#include <ctime>
using namespace std;

#define N 10000000


int GetMaxDigit(int *a, int n){
    //max_element 找到最大元素地址函数，记住是左闭右开哦
    int maxdata = *max_element(a, a + n);//找到最大元素计算其最大位数即可   
    int maxdigit = 0;                    //最大位数
    while(maxdata){
	maxdata /= 10;
	maxdigit++;
    }
    return maxdigit;
}


//基数排序
void RadixSort(int *a, int n){
    int base = 1, digit = GetMaxDigit(a, n);
    int *tmp = new int[n];                 //临时数组  
    int *count = new int[10];              //统计数组，统计某一位数字相同的个数
    int *start = new int[10];              //起始索引数组，某一位数字相同数字的第一个的位置
    
    //最大位数为多少，就循环多少次
    while(digit--){
	memset(count, 0, 10 * sizeof(int));//每一次都全初始化为0
	//不可以写sizeof(count),这是指针的大小(若为64位，则为8),和普通数组的数组名不一样
        for(int i = 0; i < n; i++){
            int index = a[i] / base % 10;  //每一位数字
            count[index]++;
        }
         
        memset(start, 0, 10 * sizeof(int));//每一次都全初始化为0
        for(int i = 1; i < 10; i++)
            start[i] = count[i - 1] + start[i - 1];

        memset(tmp, 0, n * sizeof(int));   //每一次都全初始化为0
        for(int i = 0; i < n; i++){
            int index = a[i] / base % 10;
            tmp[start[index]++] = a[i];    //某一位相同的数字放到临时数组中合适的位置
        }

        memcpy(a, tmp, n * sizeof(int));   //复制tmp中的元素到a
        base *= 10;                        //比较下一位
    }      

    delete[] tmp;                          //释放空间
    delete[] count;
    delete[] start;                     
}


void show(int *a, int n){
    for(int i = 0; i < n; i++)
	cout<<*(a + i)<<" ";
    cout<<endl;
}


int main(){
    int *a = (int*) malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) {
        a[i] = int(rand() / (float) RAND_MAX * 10);
    }

    float esp_time_cpu;
	clock_t start_cpu, stop_cpu;
    start_cpu = clock();
    RadixSort(a, N);
	stop_cpu = clock();
    esp_time_cpu = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;

	printf("The time by host:\t%f(ms)\n", esp_time_cpu);
}