set -e

#echo "3D Tests"
#----------- Radix 2

for j in 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
do
	bin/3d $j 4 4 2> /dev/null
done

#----------- Radix 4
#for j in 12 
#do
#	bin/1d $(( 4 ** $j )) 2> /dev/null
#done

#----------- Radix 8
#for j in 8
#do
#	bin/1d $(( 8 ** $j )) 2> /dev/null
#done

#holdMax=1000
#for (( j = 10 ; j <= 1000; j+=10 )) 
#do
#	./a.out 8388608 1 1 300 uniform uniform $j no 2> /dev/null     
#done
