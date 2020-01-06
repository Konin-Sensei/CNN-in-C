/* 
 * File:   generate.c
 * Author: konin
 *
 * Created on February 27, 2019, 7:41 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_LENGTH 784
#define LAYER_LENGTH 8
#define OUTPUT_LENGTH 10
#define NETWORK_LENGTH 5
double rand_double();

/*
 * 
 */
int main(int argc, char** argv) {
    FILE *weight_ptr;
    FILE *bias_ptr;
    weight_ptr = fopen("weights","w");
    bias_ptr = fopen("biases","w");
    srand(time(NULL));
    
    int i, j, k;
    for(i = 1; i < NETWORK_LENGTH; i++){
        if(i == 1){
            for(j = 0; j < LAYER_LENGTH; j++){
                for(k = 0; k < INPUT_LENGTH; k++){
                    fprintf(weight_ptr, "%.2lf ",rand_double()/100);
                }
                fprintf(weight_ptr, "\n");
                fprintf(bias_ptr, "%.2lf ", rand_double()/100);
            }
        }else if(i == NETWORK_LENGTH - 1){
            for(j = 0; j < OUTPUT_LENGTH; j++){
                for(k = 0; k < LAYER_LENGTH; k++){
                    fprintf(weight_ptr, "%.2lf ",rand_double()/100);
                }
                fprintf(weight_ptr, "\n");
                fprintf(bias_ptr, "%.2lf ", rand_double()/100);
            }
        }else{
            for(j = 0; j < LAYER_LENGTH; j++){
                for(k = 0; k < LAYER_LENGTH; k++){
                    fprintf(weight_ptr, "%.2lf ",rand_double()/100);
                }
                fprintf(weight_ptr, "\n");
                fprintf(bias_ptr, "%.2lf ", rand_double()/100);
            }
        }
    }

    return (EXIT_SUCCESS);
}
double rand_double(){
    return rand()%100 + 1;
}
