@echo off
REM
call conda activate EventSRSNN

REM
cd /c C:\code\EventSR-Project\EvSR-test2\nMnist

REM
python trainNmnist_Louck_triple.py ^
--bs 64 ^
--savepath "C:/code/EventSR-Project/EventSR-ckpt/lr_5_0.5/" ^
--epoch 30 ^
--showFreq 50 ^
--lr 0.1 ^
--cuda "0" ^
--j 4

pause
