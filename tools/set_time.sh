# 定时脚本，倒计时5小时后结束
#!/bin/bash
END=$((SECONDS+18000))  # 5 hours = 18000 seconds
while [ $SECONDS -lt $END ]; do
    REMAINING=$((END-SECONDS))
    HOURS=$((REMAINING/3600))
    MINUTES=$(( (REMAINING%3600)/60 ))
    SECONDS_LEFT=$((REMAINING%60))
    printf "\rTime remaining: %02d:%02d:%02d" $HOURS $MINUTES $SECONDS_LEFT
    sleep 1
done
echo -e "\nTime's up! Exiting."