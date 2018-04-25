import java.lang.Exception;
import java.lang.Long;
import java.lang.String;
import java.lang.System;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Reader;
import java.math.BigInteger;
import java.math.BigDecimal;
class sys2
{

    public static void simulate(InputStream incomingStream, int cacheSize, int kway, int verbose, int ic1, int ic2) throws Exception 
    {
        // See the documentation to understand what these variables mean.
        long microOpCount; // 1
        long instructionAddress;// 2
        long sourceRegister1; // 3
        long sourceRegister2; // 4
        long destinationRegister; // 5
        String conditionRegister; // 6
        String TNnotBranch; // 7
        String loadStore; // 8
        long immediate; // 9
        long addressForMemoryOp; // 10
        long fallthroughPC; // 11
        long targetAddressTakenBranch; // 12 
        String macroOperation; // 13 
        String microOperation; // 14

        long totalMicroops = 0; 
        long totalMacroops = 0;
        int k = kway;
        int order = 0;
        int blocksize = 16; //bytes
        int offsetBits = 4;
        int cache = cacheSize * 1024;
        double numOfsets = cache / (k * blocksize); // k way, 1 set has k blocks
        double indexBits = Math.log(numOfsets) / Math.log(2);

        int validBit[][] = new int [(int)numOfsets][k];
        int dirtyBit[][] = new int [(int)numOfsets][k];
        String tag[][] = new String [(int)numOfsets][k];
        int lastUsed[][] = new int [(int)numOfsets][k];
        int id = 0;

        int hit = 0;
        int penalty = 80;
        int dirtyNow = 0;
        int numRead = 0;
        int numWrite = 0;
        int numDataAccess = 0;
        int readMiss = 0;
        int writeMiss = 0;
        int dataMiss = 0;
        int dReadMiss = 0;
        int dWriteMiss = 0;
        int readBytesFromMemory = 0;
        int writeBytesToMemory = 0;
        int totalReadTime = 0;
        int totalWriteTime = 0;
        double missRate = 0;
        int lastUsedTmp = 0;
        int validBitTmp = 0;
        String lastStoredHex = "";

        for (int i = 0; i < numOfsets; i++) {
            for (int j = 0; j < k; j++) {
                validBit[i][j] = 0;
                tag[i][j] = "0";
                dirtyBit[i][j] = 0;
                lastUsed[i][j] = -1;
            }
        }

        BufferedReader r = new BufferedReader(new InputStreamReader(incomingStream));
        String line;




        while (true) {
            line = r.readLine();
            if (line == null) {
                break;
            }
            String [] tokens = line.split("\\s+");

            microOpCount = Long.parseLong(tokens[0]);
            instructionAddress = Long.parseLong(tokens[1], 16);
            sourceRegister1 = Long.parseLong(tokens[2]);
            sourceRegister2 = Long.parseLong(tokens[3]);
            destinationRegister = Long.parseLong(tokens[4]);
            conditionRegister = tokens[5];
            TNnotBranch = tokens[6];
            loadStore = tokens[7];
            immediate = Long.parseLong(tokens[8]);
            addressForMemoryOp = Long.parseLong(tokens[9], 16);
            fallthroughPC = Long.parseLong(tokens[10], 16);
            targetAddressTakenBranch = Long.parseLong(tokens[11], 16);
            macroOperation = tokens[12];
            microOperation = tokens[13];

            // For each micro-op
            totalMicroops++;
            
                // For each macro-op
            if (microOpCount == 1) {
                totalMacroops++;
            }

            if (loadStore.equals("L") || (loadStore.equals("S"))) {
                
                String BinaryFullStr = Long.toBinaryString(addressForMemoryOp);
                String BinaryOffset = BinaryFullStr.substring(BinaryFullStr.length() - offsetBits, BinaryFullStr.length());
                String BinaryIndex = BinaryFullStr.substring(BinaryFullStr.length() - offsetBits - (int)indexBits, BinaryFullStr.length() - offsetBits);
                String BinaryTag = BinaryFullStr.substring(0, BinaryFullStr.length() - offsetBits - (int)indexBits);


              /*  int decIndex = Integer.parseInt(BinaryIndex,2);
                int decTag = Integer.parseInt(BinaryTag,2);
                String hexIndex = Integer.toString(decIndex,16);
                String hexTag = Integer.toString(decTag,16);*/
                long decIndex = Long.parseLong(BinaryIndex,2);
                long decTag = Long.parseLong(BinaryTag,2);
                String hexIndex = Long.toString(decIndex,16);
                String hexTag = Long.toString(decTag,16);
                int dirtytmp = 0;
                

                loop:
                for (int b = 0; b < k; b++) {   //case 1 hit
                    if (tag[(int)decIndex][b].equals(hexTag)) {
                        id = b;
                      //  if (( order >= ic1) && (order <= ic2)) { System.out.print("   TAG   " + hexTag + "   =   " + tag[(int)decIndex][b] + "   ");}
                        dirtytmp = dirtyBit[(int)decIndex][id];
                        if (loadStore.equals("S")) {
                            dirtyBit[(int)decIndex][id] = 1;    //store so change dirty bit
                        } else {}
                            // Read = No state change                        }
                        validBitTmp = validBit[(int)decIndex][id];
                        validBit[(int)decIndex][id] = 1;
                        lastUsedTmp = lastUsed[(int)decIndex][id];
                        lastUsed[(int)decIndex][id] = order;
                        lastStoredHex = hexTag;
                        hit = 1;
                        break loop;
                    }
                } 

                int smallest = 0;
                if (hit == 0) {                   //case 2, miss
                    for (int b = 1; b < k; b++) { //Loop to find the last used block
                        if (lastUsed[(int)decIndex][smallest] > lastUsed[(int)decIndex][b]) {
                            smallest = b;
                        }       
                    }
                    id = smallest;
                    dirtytmp = dirtyBit[(int)decIndex][id];
                    validBitTmp = validBit[(int)decIndex][id];
                    validBit[(int)decIndex][id] = 1;
                    lastStoredHex = tag[(int)decIndex][id];
                    tag[(int)decIndex][id] = hexTag;
                    lastUsedTmp = lastUsed[(int)decIndex][id];
              
                    lastUsed[(int)decIndex][id] = order;

                    if (dirtytmp == 0) { //case 2a
                        if(loadStore.equals("S")) {
                            dirtyBit[(int)decIndex][id] = 1;
                        } else {
                            dirtyBit[(int)decIndex][id] = 0;
                        }
                    } else {  //case 2b
                        if(loadStore.equals("S")) {
                            dWriteMiss++;
                            dirtyBit[(int)decIndex][id] = 1;
                        } else {
                            dirtyBit[(int)decIndex][id] = 0;
                            dReadMiss++;
                        }
                    }
                }
                    
                 else {
                    if (loadStore.equals("S")) {
                            totalWriteTime+=1;
                        } else {
                            totalReadTime+=1;
                        }
                }

                //Calculation and printing
                if (( order >= ic1) && (order <= ic2) && (verbose == 1)) {
                    System.out.print(order + " " + tokens[9] + " " + hexIndex +" " + hexTag + " " + validBitTmp + " " + id + " ");
                    if (lastUsedTmp == -1) {
                        System.out.print("0 " + lastStoredHex + " " + dirtytmp + " " + hit);
                    } else {
                        System.out.print(lastUsedTmp + " " + lastStoredHex + " " + dirtytmp + " " + hit);
                    }
                }

                if (hit == 1) {
                    if (( order >= ic1) && (order <= ic2) && (verbose == 1)) {
                        System.out.println(" 1");
                    }
                } else if (dirtytmp == 1){
                    if (( order >= ic1) && (order <= ic2) && (verbose == 1)) {
                        System.out.println(" 2b");
                    }
                    if (loadStore.equals("L")) {
                        readMiss++;
                        totalReadTime = totalReadTime + 1 + (2*penalty);
                    } else {
                        writeMiss++;
                        totalWriteTime = totalWriteTime + 1 + (2*penalty);
                        }
                    } 

                else {
                    if (( order >= ic1) && (order <= ic2) && (verbose == 1)) {
                        System.out.println(" 2a");
                    }
                    if (loadStore.equals("L")) {
                        readMiss++;
                        totalReadTime = totalReadTime + 1 + penalty;  
                        } else {
                            writeMiss++;
                            totalWriteTime = totalWriteTime + 1 + penalty;
                            
                        }
                    }
                 
                order++;

                if (loadStore.equals("L")) {numRead++;} else {numWrite++;}

            }
            hit = 0;
        }
        //output variable
        numDataAccess = numRead + numWrite;
        dataMiss = readMiss+writeMiss;
        readBytesFromMemory = 16 * (readMiss+writeMiss); 
        writeBytesToMemory = 16 * (dReadMiss+dWriteMiss);
        missRate = (double)dataMiss/ numDataAccess;

        System.out.println("number of data reads: " + numRead +
                            "\nnumber of data writes: " + numWrite +
                            "\nnumber of data accesses: " + numDataAccess +
                            "\n\nnumber of data read misses: " + readMiss +
                            "\nnumber of data write misses: " + writeMiss +
                            "\nnumber of data misses: " + (readMiss + writeMiss)+
                            "\n\nnumber of dirty data read misses: " + dReadMiss +
                            "\nnumber of dirty write misses: " + dWriteMiss +
                            "\n\nnumber of bytes read from memory: " + readBytesFromMemory +
                            "\nnumber of bytes written to memory: " + writeBytesToMemory +
                            "\n\nthe total access time (in cycles) for reads: " + totalReadTime +
                            "\nthe total access time (in cycles) for writes: " + totalWriteTime +
                            "\nthe total access time for reads and writes: " + (totalWriteTime+totalReadTime) +
                            "\n\nthe overall data cache miss rate: " + missRate); 
    }


    public static void main(String[] args) throws Exception
    {
        InputStream inputStream = System.in;
        int cacheSize = 0;
        int verboseMode = 0;
        int sets = 0;
        int ref1 = 0;
        int ref2 = 0;

        if (args.length == 6) {
            inputStream = new FileInputStream(args[0]);
            cacheSize = Integer.parseInt(args[1]);
            sets = Integer.parseInt(args[2]);
            ref1 = Integer.parseInt(args[4]);
            ref2 = Integer.parseInt(args[5]);
            if (args[3].equals("-v")) {
                System.out.println("Verbose Mode On");
                verboseMode = 1;
            }

            if (ref1 >= ref2) {
                System.out.println("Error, IC2 has to be smaller than IC1");
                System.exit(1);
            }
        } else {
            System.out.println("Sorry, please re-enter the parameter in the way as java sys2 [tracefile.xac] [cacheSize] [k-way-sets] [verboseMode] [IC1] [IC2]\ne.g. {java sys2 gcc.xac 2 2 -v 100 200} for verbose mode, \n{java sys2 gcc.xac 2 2 [anything] 100 200} for normal mode");
            System.exit(1);
        }


        sys2.simulate(inputStream, cacheSize, sets, verboseMode, ref1, ref2);
        //long test = Long.parseLong("aa",16);
        //System.out.println(Long.toString(test,2));
    }
}
