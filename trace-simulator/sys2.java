/**
* sys2 is a Branch Prediction Simulator program (basic 2-bit predictor with branch target buffer)
* sys2 read the trace file and return a table of details including:
*       1.the total number of conditional branches
*       2.the number of forward branches
*       3.the number of backward branches
*       4.the number of forward taken branches
*       5.the number of backward taken branches
*       6.the number of mispredicted branches
*       7.the misprediction rate for all branches (# mispredictions / # branches)
*	8.the number of BTB misses
*       9.the BTB miss rate (# BTB misses / # BTB accesses)
* Compilation: javac sys2.java
* Run: java sys2 xxx.trace N M [-v] (-v for verbose mode to list more detail of trace table) (Please put the test trace file and sys2 in same directory)
* @author Peitong Shi
* @since 2018-04-02
*
**/






import java.math.BigInteger;
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

class sys2
{
    
    public static void simulate(InputStream incomingStream, int numPB, int numBTB, int verboseMode) throws Exception 
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
        int NOB = 0; // numbers of branches
 	int FB = 0; // Forward branches
	int FTB = 0; // Forward taken branches
	int BB = 0;  // Backward branches
	int BTB = 0; // Backward taken branches
	int MP = 0; // Number of mispredictions
 	int missedBTB = 0; //This is numer of "missed Branch Buffer Target"
	int missedBTBRate = 0;
	//SYS2
	
	int PBbits = (int)(Math.log(numPB)/Math.log(2));
	int BTBbits = (int)(Math.log(numBTB)/Math.log(2));

        int PBuffer[] = new int [numPB];
	int BTBuffer[] = new int [numBTB];
	String BTBtag [] = new String [numBTB];
	for (int i = 0; i < numPB; i++) {
	    PBuffer[i] = 1;
	}		
	for (int j = 0; j < numBTB;j++) {
	    BTBuffer[j] = 0;
	    BTBtag[j] = "";
	}
	int BTBhit = 0;
	int BTBmiss = 0;
	int BTBaccess = 0;	
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

	    //System.out.println(TNnotBranch);
	    //System.out.println(instructionAddress);
            if (TNnotBranch.equals("N") || TNnotBranch.equals("T")) { 
		String BinaryFullStr =  Long.toBinaryString(instructionAddress);
		String BinaryPBStr = BinaryFullStr.substring(BinaryFullStr.length() - PBbits, BinaryFullStr.length());
		String BinaryBTBStr = BinaryFullStr.substring(BinaryFullStr.length() - BTBbits, BinaryFullStr.length());
		String BinaryTag = BinaryFullStr.substring(0, BinaryFullStr.length() - BTBbits);
		BigInteger bi = new BigInteger(BinaryTag, 2);
		String Taghex = bi.toString(16);
		int PBindex = Integer.parseInt(BinaryPBStr,2);	
		int BTBindex = Integer.parseInt(BinaryBTBStr,2);
		boolean taken = false;
		boolean predictTaken = false;

		//System.out.println(BinaryFullStr + "\n" + BinaryPBStr + "\n" + BinaryBTBStr +"\n")
		if (verboseMode == 1) {
			System.out.print(NOB + " " + PBindex + " " + PBuffer[PBindex] + " " );
		}

		if (TNnotBranch.equals("T")) {taken = true; /*System.out.print("Taken ");*/ } 
	/*	if (BTBtag[BTBindex].length() == 0) {BTBtag[BTBindex] = Taghex; /*System.out.print("Empty at " + BTBindex + "Tag " + Taghex+ " ");*/
		
		if (PBuffer[PBindex] >=2) {
		    predictTaken = true;
		}

               /* if (taken && predictTaken) {
                     BTBuffer[BTBindex] = 1;
                     BTBtag[BTBindex] = Taghex;
			System.out.print(" now ");
                }*/


               /* if (BTBtag[BTBindex].equals("")) { 
		    BTBtag[BTBindex] = Taghex;
		} else if (taken && predictTaken) {
                     BTBuffer[BTBindex] = 1;
                     BTBtag[BTBindex] = Taghex;
                        BTBmiss++;
               }*/
		if (BTBtag[BTBindex].equals("")) { BTBtag[BTBindex] = Taghex;}	 //if the BTB[index] is empty, store the first Tag we meet
	
	        if (predictTaken) {              
		    if (BTBtag[BTBindex].equals(Taghex) && BTBuffer[BTBindex] == 1) { BTBhit++;} else {BTBmiss++;} //if valid bit is 1, and tag is the same, HIT
		    if (taken) { BTBtag[BTBindex] = Taghex; BTBuffer[BTBindex] = 1;}
                }
		
                

	/*	if (predictTaken) {
		    if (BTBtag[BTBindex].equals(Taghex) && BTBuffer[BTBindex] ==1) {
		    	BTBhit++;
		    }else {BTBmiss++;}
		}*/
		
       


		if (instructionAddress < targetAddressTakenBranch) { 
			FB++;
			if (taken) { 
			    FTB++;
	         	    if (PBuffer[PBindex] < 3) { PBuffer[PBindex]++; }
			    
			} 
			else { if(PBuffer[PBindex] > 0) { PBuffer[PBindex]--; }}
		} 
		else { 
			BB++;
			//System.out.println("Backward Branch");
			if (taken) { 
			    BTB++;
			    if (PBuffer[PBindex] < 3) { PBuffer[PBindex]++; }
			}
			else {
			    if (PBuffer[PBindex] > 0) { PBuffer[PBindex]--; }
			}
		}
		if (predictTaken != taken) { MP++;}
		NOB++;
		BTBaccess = BTBhit + BTBmiss;

		if (verboseMode == 1) {
	    	    System.out.println(PBuffer[PBindex] + " " + BTBindex + " " + Taghex + " " + BTBaccess + " " + BTBmiss);
	   	}
	  	// System.out.println("Current hextag: " + BTBtag[BTBindex] + "Current VB: " + BTBuffer[BTBindex]);
	   // System.out.println("--------------");
           }
	}
        
	System.out.println("Number of branches = " + NOB 
			    + "\nNumber of forward branches = " + FB
			    + "\nNumber of forward taken branches = " + FTB
			    + "\nNumber of backward branches = " + BB
			    + "\nNumber of backward taken branches = " + BTB
			    + "\nNumber of misprediction = " + MP
			    + "\nMisprediction rate = " + ((double)MP/(NOB))
			    + "\nNumber of BTB miss = " + BTBmiss + " " + (double) BTBmiss / BTBaccess);
    }
    
    public static void main(String[] args) throws Exception

    {
        InputStream inputStream = System.in;
//	inputStream = new FileInputStream(args[0]);
	if (args.length >= 3) {
       	    int NoePB = Integer.parseInt(args[1]);
	    int NoeBTB = Integer.parseInt(args[2]);
	  //  int numOfBitsPB;
          //  int numOfBitsBTB;
	    boolean check1 = NoePB > 0 && ((NoePB & (NoePB -1)) == 0);
	    boolean check2 = NoeBTB > 0 && ((NoeBTB & (NoeBTB - 1)) == 0);
    
            inputStream = new FileInputStream(args[0]);
	    if (check1 && check2){
		inputStream = new FileInputStream(args[0]);
		if (args.length == 4 && args[3].equals("-v")) {
		    System.out.println("Verbose Mode On");
		    sys2.simulate(inputStream, NoePB, NoeBTB, 1);
		}else {
                //numOfBitsPB = (int) (Math.log(NoePB)/Math.log(2));
		//numOfBitsBTB = (int) (Math.log(NoeBTB)/Math.log(2));
		    System.out.println("Verbose Mode Off, use \"java sys xxx.trace N M -v \"to enable Verbose Mode to see the lists of integer for simulation");
		    sys2.simulate(inputStream,NoePB, NoeBTB, 0);
		}		
	    } else {
	        System.out.println("Numbers of entries of Predict Buffer and Branch Taken Buffer has to be power of 2");
	    }
        } 
	 else {
	    System.out.println("Error on entering parameter, please check and re-run the project");
       }

	//long test = Long.parseLong("aa",16);
	//System.out.println(Long.toString(test,2));
	
   }
}
