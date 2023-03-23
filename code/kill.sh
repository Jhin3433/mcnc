# ps -aux |grep SimCSE|grep Sl|awk '{print $2}'|xargs kill -9
ps -aux |grep SimCSE|awk '{print $2}'|xargs kill -9