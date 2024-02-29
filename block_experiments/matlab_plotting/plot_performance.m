load('../data/performance.mat')

close all
markers = {'o','*','s','^'};
colors = {'#648FFF','#DC267F','#785EF0','#FE6100'};
methods = {'BlockRPC','Accelerated'};
qualities = {'time', 'error'};

for quality_idx = 1:length(qualities)
    f = figure(quality_idx);
    f.Position = [100 100 1000 400];
    quality = qualities{quality_idx};
    baseline = eval(sprintf('RPCholesky_%s',quality));
    [~,idx] = sort(eval(sprintf('BlockRPC_%s',quality)) ./ baseline);
    if quality_idx == 1
        idx = idx(end:-1:1);
    end
    for method_idx = 1:length(methods)
        method = methods{method_idx};
        data = eval(sprintf('%s_%s',method,quality));
        if quality_idx == 1
            to_plot = baseline(idx) ./ data(idx);
        elseif quality_idx == 2
            to_plot = data(idx) ./ baseline(idx);
        end
        plot(1:length(data),to_plot,'Marker',...
            markers{method_idx},'MarkerFaceColor',colors{method_idx},...
            'Color',colors{method_idx},'LineWidth',2,...
            'MarkerSize',14,'LineStyle','none')
        hold on
    end

    if quality_idx == 1
        xlabel('Problem (sorted by block RPCholesky speedup)')
        ylabel('Speedup $\frac{\rm Method\: time}{\rm RPCholesky\: time}$')
        axis([0 length(baseline) 0 50])
    elseif quality_idx == 2
        set(gca,'YScale','log')
        xlabel('Problem (sorted by block RPCholesky error ratio)')
        ylabel('Error ratio $\frac{\rm Method\: error}{\rm RPCholesky\: error}$')
        legend({'Block RPCholesky','Accelerated RPCholesky'},...
            'Location','northwest')
        axis([0 length(baseline) -Inf Inf])
    end
    saveas(gcf,sprintf('../figs/performance_%s.png',quality))
    saveas(gcf,sprintf('../figs/performance_%s.fig',quality))
end