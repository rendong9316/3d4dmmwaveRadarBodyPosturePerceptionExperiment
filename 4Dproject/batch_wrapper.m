%% batch_wrapper.m - Run run_4Dpointcloud.m for multiple scenarios
scenarios = {'CCdata_walk_0001','CCdata_jump_0001','CCdata_stand_0001'};
for s = 1:length(scenarios)
    scenario = scenarios{s};
    fprintf('=== %s ===\n', scenario);
    run('run_4Dpointcloud.m');
end
fprintf('Done\n');
