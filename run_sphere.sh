
#sbatch start.sh "clean" 0.1 10
#sbatch start.sh "truemax" 0.1 10

for i in 1 5 10 50 100
do
	for j in 0.1 0.01
	do
		sbatch start.sh "adv" $j $i
	done
done
