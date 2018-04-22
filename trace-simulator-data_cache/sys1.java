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
class sys1
{

    public static void simulate(InputStream incomingStream, int cacheSize, int verbose, int ic1, int ic2) throws Exception 
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
        int order = 0;
        int blocksize = 16; //bytes
        int offsetBits = 4;
        int cache = cacheSize * 1024;
        double numOfBlocks = cache / blocksize;
        double indexBits = Math.log(numOfBlocks) / Math.log(2);

        int validBit[] = new int [(int)numOfBlocks];
        int dirtyBit[] = new int [(int)numOfBlocks];
        String tag[] = new String [(int)numOfBlocks];

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

        for (int i = 0; i < numOfBlocks; i++) {
            validBit[i] = 0;
            tag[i] = "0";
            dirtyBit[i] = 0;
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


                if (tag[(int)decIndex].equals(hexTag)) {
                   if (validBit[(int)decIndex] == 0) {
                        validBit[(int)decIndex] = 1;
                    }
                    hit = 1;
                }



           //     String hexIndex = Integer.toString(decIndex,10);
             //   String hexTag = Integer.toString(decTag,10);

                if (( order >= ic1) && (order <= ic2)) {
                    System.out.print(order + " " + tokens[9] + " " + hexIndex +" " + hexTag + " " + validBit[(int)decIndex] + " " + tag[(int)decIndex] + " " + dirtyBit[(int)decIndex] + " " + hit);
                }
                if (hit == 1) {
                    if (( order >= ic1) && (order <= ic2)) {
                        System.out.println(" 1");
                    }
                    if(loadStore.equals("L")) {
                        totalReadTime+=1;
                    } else {
                        totalWriteTime+=1;
                    }
                }  else if (dirtyBit[(int)decIndex] == 1){
                    if (( order >= ic1) && (order <= ic2)) {
                        System.out.println(" 2b");
                    }
                    if (loadStore.equals("L")) {
                        totalReadTime = totalReadTime + 1 + (2*penalty);
                        readMiss++;
                        dReadMiss++;
                    } else {
                        totalWriteTime = totalWriteTime + 1 + (2*penalty);
                        writeMiss++;
                        dWriteMiss++;
                        }
                    } else {
                        if (( order >= ic1) && (order <= ic2)) {
                           System.out.println(" 2a");
                        }
                        if (loadStore.equals("L")) {
                            totalReadTime = totalReadTime + 1 + penalty;
                            readMiss++;
                        } else {
                            totalWriteTime = totalWriteTime + 1 + penalty;
                            writeMiss++;
                        }
                    }
                

                if ((tag[(int)decIndex] != hexTag) && (hit == 0)) {
                    tag[(int)decIndex] = hexTag;
                    dirtyBit[(int)decIndex] = 0;
                }
                if (loadStore.equals("S")) {
                    dirtyBit[(int)decIndex] = 1;
                }
                order++;

                if (loadStore.equals("L")) {numRead++;} else {numWrite++;}
            }
            hit = 0;
        }
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
                            "\n\nthe overall data cache miss rate: " + missRate); 
    }


    public static void main(String[] args) throws Exception
    {
        InputStream inputStream = System.in;
        int cacheSize = 0;
        int verboseMode = 0;
        int ref1 = 0;
        int ref2 = 0;

        if (args.length == 5) {
            inputStream = new FileInputStream(args[0]);
            cacheSize = Integer.parseInt(args[1]);
            ref1 = Integer.parseInt(args[3]);
            ref2 = Integer.parseInt(args[4]);
            if (args[2].equals("-v")) {
                System.out.println("Verbose Mode On");
                verboseMode = 1;
            }

            if (ref1 >= ref2) {
                System.out.println("Error, IC2 has to be smaller than IC1");
                System.exit(1);
            }
        } else {
            System.out.println("Sorry, please re-enter the parameter in the way as java sys1 [tracefile.xac] [cacheSize] [verboseMode] [IC1] [IC2]\ne.g. {java sys1 gcc.xac 4 -v 100 200}");
            System.exit(1);
        }


        sys1.simulate(inputStream, cacheSize, verboseMode, ref1, ref2);
    	//long test = Long.parseLong("aa",16);
    	//System.out.println(Long.toString(test,2));
    }
}
