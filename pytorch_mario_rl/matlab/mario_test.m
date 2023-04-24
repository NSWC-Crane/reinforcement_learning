format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[startpath,  filename, ext] = fileparts(full_path);
plot_num = 1;

%% load in the data

data_directory = 'C:\Projects\data\mario\emerson_2023-04-11T19-37-51';

[states, actions] = read_mario_results(data_directory);

%% future analysis

