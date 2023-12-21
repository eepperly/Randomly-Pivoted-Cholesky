load('../data/performance.mat')

markers = {'o','*','s','^'};
colors = {'#648FFF','#785EF0','#DC267F','#FE6100'};
methods = {'BlockRPC','Accelerated','RPCholesky'};
qualities = {'time', 'error'};

for quality_idx = 1:length(qualities)
    figure(quality_idx)
    quality = qualities{quality_idx};
    for method_idx = 1:length(methods)
        method = methods{method_idx};
        data = eval(sprintf('%s_%s',method,quality));
        plot((1:length(data))/length(data),sort(data),'Marker',...
            markers{method_idx},'MarkerFaceColor',colors{method_idx},...
            'Color',colors{method_idx},'LineWidth',2,...
            'MarkerSize',10)
        hold on
    end

    xlabel('Fraction of problems')
    if quality_idx == 1
        ylabel('Time (sec)')
    elseif quality_idx == 2
        set(gca,'YScale','log')
        ylabel('Error ratio')
    end
end