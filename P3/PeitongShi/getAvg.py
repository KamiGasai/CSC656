import sys, string

file = open("output.txt", "r")
vals = list()
for line in file:
  if(line[0] == "R"):
    particleAmt = line.split(":")[1]
    particleAmt = particleAmt.strip("\n")
    print "-" * 10  +  "Particle Amount: " + particleAmt + " " +  "-"*10 + "\n"
  if(line[0] == "T"):
    threadCount = line.split(":")[1]
    threadCount = threadCount.strip("\n")
    print "Thread Count: " + threadCount
  if(line[0] == "E"):
    vals.append( float( line.split("=")[1] ) )
    if(len(vals) == 5):
      vals.remove(max(vals))
      vals.remove(min(vals))
      sum = 0.0
      for val in vals:
        sum+=val
      avg = sum/len(vals)
      print("Average is: %.6f\n" % avg)
      vals = list()

    
    


