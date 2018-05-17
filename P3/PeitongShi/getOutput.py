import sys, string, os, fileinput

particleAmounts = ['1024','8192','32768' ]
threadSizes = ['4', '16', '64']

def changeNPDirective( newValue ):
  for line in fileinput.input("findRedsDriver.cu", inplace=True):
    if "#define" in line and "NUMPARTICLES" in line:
      print "#define NUMPARTICLES " + newValue 
    else:
      print line,
fileinput.close()

def changeTPBDirective( newValue):
  for line in fileinput.input("findRedsDriver.cu", inplace=True):
    if "#define" in line and "THREADSPERBLOCK" in line:
      print "#define THREADSPERBLOCK " + newValue 
    else:
      print line,
fileinput.close()

for particleAmt in particleAmounts:
  changeNPDirective(particleAmt)
  os.system("echo RUNNING FOR Particle: " + particleAmt + " >> output.txt")
  for threads in threadSizes:
    changeTPBDirective(threads)
    os.system("echo THREAD COUNT: " + threads + " >> output.txt")
    os.system("nvcc findRedsDriver.cu -Wno-deprecated-gpu-targets")
    for _ in range(0, 5):  
      os.system("./a.out " + particleAmt + " " + threads + " >> output.txt")

os.system("rm a.out")
