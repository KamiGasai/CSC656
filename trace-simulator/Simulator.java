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

class Simulator
{
    
    public static void simulate(InputStream incomingStream, PrintStream outputStream) throws Exception 
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

        BufferedReader r = new BufferedReader(new InputStreamReader(incomingStream));
        String line;
        
        outputStream.format("Processing trace...\n");
        
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
		NOB++;
		if (instructionAddress < targetAddressTakenBranch) { 
			FB++;
			if (TNnotBranch.equals("T")) { FTB++;MP++;}
		} 
		else { 
			BB++;
			if (TNnotBranch.equals("T")) { BTB++;}
			else { MP++;}
		}

	    }
        }
        
        outputStream.format("Processed %d trace records.\n", totalMicroops);
        
        outputStream.format("Micro-ops: %d\n", totalMicroops);
        outputStream.format("Macro-ops: %d\n", totalMacroops);
	System.out.println("Number of branches = " + NOB 
			    + "\nNumber of forward branches = " + FB
			    + "\nNumber of forward taken branches = " + FTB
			    + "\nNumber of backward branches = " + BB
			    + "\nNumber of backward taken branches = " + BTB
			    + "\nNumber of misprediction = " + MP
			    + "\nMisprediction rate = " + ((double)MP/(NOB)));
    }
    
    public static void main(String[] args) throws Exception
    {
        InputStream inputStream = System.in;
        PrintStream outputStream = System.out;
        
        if (args.length >= 1) {
            inputStream = new FileInputStream(args[0]);
        }
        
        if (args.length >= 2) {
            outputStream = new PrintStream(args[1]);
        }
        
        Simulator.simulate(inputStream, outputStream);
	//long test = Long.parseLong("aa",16);
	//System.out.println(Long.toString(test,2));
	
    }
}
