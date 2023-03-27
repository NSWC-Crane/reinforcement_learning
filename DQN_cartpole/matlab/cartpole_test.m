format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[startpath,  filename, ext] = fileparts(full_path);
plot_num = 1;

line_width = 1;
%% get the data

file_filter = {'*.csv','CSV Files';'*.*','All Files' };

[input_file, input_path] = uigetfile(file_filter, 'Select Video File', startpath);
if(input_path == 0)
    return;
end

% read in the csv data
q_obs = readmatrix(fullfile(input_path, input_file));

% separate out into state observations and actions
A0 = q_obs(q_obs(:,5)==0,:);
A1 = q_obs(q_obs(:,5)==1,:);

states = q_obs(:, 1:4);
actions = q_obs(:, 5);

%% just plot the two main variables
cm = [1 0 0; 0 0 1];

figure
set(gcf,'position',([50,50,1400,800]),'color','w')

hold on
box on
grid on

scatter(A0(:,3), A0(:,4), 5, 'r', 'filled');
scatter(A1(:,3), A1(:,4), 5, 'b', 'filled');

set(gca,'fontweight','bold','FontSize', 13);
xlabel('Cart Velocity', 'fontweight', 'bold', 'FontSize', 13);
ylabel('Pole Angle', 'fontweight', 'bold', 'FontSize', 13);

title('Cartpole Action vs. State Space', 'fontweight', 'bold', 'FontSize', 14);

legend('Action: 0', 'Action: 1')

% view(90,0)

%% train on the data

[trained_model, validation_accuracy] = train_cartpole_tree(states, actions);

view(trained_model.ClassificationTree,'Mode','graph')

disp(trained_model.HowToPredict);

return;
%% or bring up the classification learner app

%select states as the xxx and actions as the response
classificationLearner;






