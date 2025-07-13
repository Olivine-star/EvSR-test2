@echo off
REM
call conda activate EventSRSNN

REM
cd  C:\code\EventSR-Project\EvSR-test2\nCifar10

REM
python trainNcifarSR_base.py ^
--bs 8 ^
--savepath "C:/code/EventSR-Project/EventSR-ckpt/ckpt-nCifar10/baseline" ^
--epoch 30 ^
--showFreq 10 ^
--lr 0.1 ^
--cuda "0" ^
--j 8

pause
