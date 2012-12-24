set -e

echo "3D Tests Horizontal"
for j in 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
do
	bin/3d $j 4 4 2> /dev/null
done

echo "3D Tests Vertical"
for j in 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
do
	bin/3d 4 $j 4 2> /dev/null
done

echo "3D Tests Planar"
for j in 4 8 16 32 64 128 256 512 1024 2048 
do
	bin/3d $j $j 4 2> /dev/null
done

echo "3D Tests Cubic"
for j in 4 8 16 32 64 
do
	bin/3d $j $j $j 2> /dev/null
done

