@echo off
REM
call conda activate EventSRSNN

REM
cd /d D:\VSCodeProject\EventSR\SR-ES1\test2\EventSR-main\nMnist

REM
python trainNmnist_1.py ^
--bs 64 ^
--savepath "D:/PycharmProjects/EventSR-ckpt/ckpt5/" ^
--epoch 30 ^
--showFreq 50 ^
--lr 0.1 ^
--cuda "0" ^
--j 4

pause
