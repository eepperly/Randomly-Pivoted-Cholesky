load('../data/performance.mat')

close all
markers = {'o','*','s','^'};
colors = {'#648FFF','#785EF0','#DC267F','#FE6100'};
methods = {'BlockRPC','Accelerated','RPCholesky'};
qualities = {'time', 'error'};

for quality_idx = 1:length(qualities)
    f = figure(quality_idx);
    f.Position = [100 100 1000 400];
    quality = qualities{quality_idx};
    if quality_idx == 1
        [~,idx] = sort(eval(sprintf('RPCholesky_%s',quality)));
    elseif quality_idx == 2
        [~,idx] = sort(eval(sprintf('BlockRPC_%s',quality)));
    end
    for method_idx = 1:length(methods)
        method = methods{method_idx};
        data = eval(sprintf('%s_%s',method,quality));
        plot(1:length(data),data(idx),'Marker',...
            markers{method_idx},'MarkerFaceColor',colors{method_idx},...
            'Color',colors{method_idx},'LineWidth',2,...
            'MarkerSize',10,'LineStyle','none')
        hold on
    end

    set(gca,'YScale','log')
    if quality_idx == 1
        xlabel('Problem (sorted by RPCholesky runtime)')
        ylabel('Time (sec)')
    elseif quality_idx == 2
        xlabel('Problem (sorted by block RPCholesky error)')
        ylabel('Error ratio')
        legend({'Block RPCholesky','Accelerated RPCholesky',...
            'RPCholesky'},'Location','northwest')
    end
end