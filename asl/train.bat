@echo off
REM
call conda activate EventSRSNN

REM
cd  D:\VSCodeProject\EventSR\SR-ES1\test2\EvSR-test2\asl

REM
python trainAsl_base.py ^
--bs 8 ^
--savepath "D:/PycharmProjects/EventSR-ckpt/Asl/baseline/baseline-ckpt---" ^
--epoch 30 ^
--showFreq 50 ^
--lr 0.1 ^
--cuda "0" ^
--j 4

pause
