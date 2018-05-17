Instruction of how to compile and run part A and B

Both files need to be run in the shell of tiger.sfsu.edu

Part A

    1. Do "source ~whsu/lees.bash_profile" to run the provided bash script to set up your paths
    
    2. Compilation:
        Navigate to the directory where your code and data live. Suppose your CUDA source file is sr.cu. Compile it ([other flags] are additional flags that you may need):  "nvcc sr.cu -o sr [other flags]"  "e.g. nvcc MaxCol.cu -o MaxCol"
        
    3. Run: MaxCol 1024 128 (matix of 1024 row/column and 128 threads per block)



Part B:

     1. Do "source ~whsu/lees.bash_profile" to run the provided bash script to set up your paths
     
     2. Compilation & Run 
         Here, used python script files by Nemi and Alvin. 
         2.1 "python getOutput.py" for keep compiling findRedsDrive.cu and
            run ./a.out [practical amount] [thread_size]. An output.txt file will be created for data record
         2.2 "python getAvg.py > resultTable.txt" this will take output.txt as input of parameter, and return a result table
         
