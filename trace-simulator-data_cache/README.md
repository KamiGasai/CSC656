# Project 2
## Data cache trace simulator 

### There are two system in this project (sys1, and sys2)
### sys1: Direct-mapped data cache simulator
### sys2: K-way set cache simulator . 

### Compilation: 
#### sys1: javac sys1.java
#### sys2: javac sys2.java . 

### Run:
#### sys1: java sys1 [trace_file_path] [cache_size] [verboseMode] [IC1] [IC2]
####          e.g. "java sys1 gcc.xac 2 -v 0 100" to trace with verbose mode, and only prints out L/S trace from [IC1] to [IC2]
####          e.g. "java sys1 gcc.xac 2 [anything] 0 100" to do same thing without verbose mode . 


#### sys2: java sys2 [trace_file_path] [cache_size] [k-way-set] [verboseMode] [IC1] [IC2]
####          e.g. "java sys2 gcc.xac 2 2 -v 0 100" to trace with verbose mode, and only prints out L/S trace from [IC1] to [IC2]
####          e.g. "java sys2 gcc.xac 2 2 [anything] 0 100" to do same thing without verbose mode . 

#### _Make sure to have something for verbose mode to run the project successfully, you have to input 5/6 args for sys1/sys2_ . 


### More:
#### Cache size are in KB, blocksize are set to be 16 bytes.
