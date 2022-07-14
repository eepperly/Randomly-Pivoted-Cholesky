matrix_names = {'smile', 'outliers'};
method_names = {'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
method_names_for_plot = {'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
last_idx = 5;
ks = 20:20:(20*last_idx);
specs = {'-', '--', ':', '*-', '-.'};

for i = 1:length(matrix_names)
    matrix_name = matrix_names{i};
    for j = 1:length(method_names)
        method_name = method_names{j};
        load(sprintf('../data/%s_%s.mat', matrix_name, method_name));
        
        for k = 1:2
            if k == 1
                errors = trace_norm_errors(1:last_idx,:);
            else
                errors = spectral_norm_errors(1:last_idx,:);
            end
            
            figure(2*i+1-k)
            if ~strcmp(method_name, 'Greedy')
                errorbar(ks, mean(errors, 2), std(errors, 1, 2),...
                    specs{j}, 'LineWidth', 2')
            else
                plot(ks, mean(errors, 2), specs{j}, 'LineWidth', 2)
            end
            hold on
        end
    end
    
    for k = 1:2
        figure(2*i+1-k)
        if strcmp(matrix_name, 'smile')
            legend(method_names_for_plot, 'location', 'best','FontSize',26)
        end
        xlabel('$k$','FontSize', 26)
        if k == 1
            ylabel('Relative Trace-Norm Error', 'FontSize', 26)
            norm_name = 'trace';
        else
            ylabel('Relatieve Spectral-Norm Error', 'FontSize', 26)
            norm_name = 'spectral';
        end
        set(gca, 'YScale', 'log')
        if 2*i+1-k == 4 && strcmp(matrix_name, 'outliers')
            axis([-Inf Inf -Inf 1e0])
        end
        ax = gca; ax.FontSize = 20; 
        saveas(gcf, sprintf('../figs/%s_%s.png', matrix_name, norm_name));
    end
end

