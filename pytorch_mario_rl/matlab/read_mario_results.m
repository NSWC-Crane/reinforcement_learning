function [states, actions] = read_mario_results(data_directory)

    file_type = '*.mat';
    listing = dir(strcat(data_directory, filesep, file_type));
    
    states = [];
    actions = [];
    
    
    for idx=1:numel(listing)
%         [data] = read_mario_matfile(fullfile(listing(idx).folder, listing(idx).name));
        data = load(fullfile(listing(idx).folder, listing(idx).name));
        
        states = cat(1, states, double(data.states));
        actions = cat(1, actions, double(data.actions(:)));
        
    end
    
    
    
    
end
