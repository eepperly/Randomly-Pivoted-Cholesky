matrix_names = {'smile', 'spiral'};
method_names = {'RPCholesky', 'Greedy', 'Uniform', 'RLS'};

close all

for i = 1:length(matrix_names)
    matrix_name = matrix_names{i};
    if strcmp(matrix_name, 'smile')
        markersize = 40;
        alpha = 0.04;
    else
        markersize = 100;
        alpha = 0.2;
    end
    for j = 1:length(method_names)
        figure((i-1)*length(method_names) + j)
        method_name = method_names{j};
        if strcmp(method_name, 'RPCholesky')
            mycolor = [0.4660 0.6740 0.1880];
        elseif strcmp(method_name, 'RLS')
            mycolor = [0.8500 0.3250 0.0980];
        elseif strcmp(method_name, 'Greedy')
            mycolor = [0.4940 0.1840 0.5560];
        else
            mycolor = [0.9290 0.6940 0.1250];
        end
        load(sprintf('../data/%s_%s_picked.mat', matrix_name, method_name))
        picked=ismember(1:size(X,1),picked);
        not_chosen = X(~picked,:);
        chosen = X(picked,:);
        scatter(not_chosen(:,1),not_chosen(:,2),'filled',...
            'MarkerFaceAlpha',alpha,'SizeData',markersize,'CData',[0 0 0])
        hold on
        scatter(chosen(:,1),chosen(:,2),'filled','Marker','p',...
            'SizeData',1800,'CData',mycolor)
        if strcmp(matrix_name, 'outliers')
            axis([-250,250,-250,250])
            % axis([-0.6,0.6,-0.6,0.6])
        end
        axis square
        set(gca,'visible','off')
        saveas(gcf, sprintf('../figs/%s_%s_chosen.png', matrix_name,...
            method_name))
    end
end