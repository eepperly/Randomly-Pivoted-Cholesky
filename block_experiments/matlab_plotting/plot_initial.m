load('../data/initial_compare.mat')

close all
markers = {'s','*','o','^'};
colors = {'#785EF0','#DC267F','#648FFF','#FE6100'};
styles = {'--','-',':'};
methods = {'Basic','Accel','Block'};
qualities = {'times', 'errs'};

for quality_idx = 1:length(qualities)
    f = figure(quality_idx);
    quality = qualities{quality_idx};
    results = {};
    for method_idx = 1:length(methods)
        method = methods{method_idx};
        data = eval(sprintf('%s_%s',method,quality));
        plot(200:200:1000,data(2:end),'Marker',...
            markers{method_idx},'MarkerFaceColor',colors{method_idx},...
            'Color',colors{method_idx},'LineWidth',2,...
            'MarkerSize',14,'LineStyle',styles{method_idx})
        hold on
    end

    xlabel('Rank $k$')
    if quality_idx == 1
        ylabel('Time (sec)')
        legend({'RPCholesky','Accelerated RPCholesky',...
            'Block RPCholesky'},'Location','northwest')
    elseif quality_idx == 2
        ylabel('Relative trace error')
        set(gca, 'YScale', 'log')
    end
    saveas(gcf,sprintf('../figs/initial_%s.png',quality))
    saveas(gcf,sprintf('../figs/initial_%s.fig',quality))
end

load('../data/points.mat')

figure(3)
markersize = 40;
alpha = 0.04;
scatter(X(:,1),X(:,2),'filled','MarkerFaceAlpha',alpha,'SizeData',...
    markersize,'CData',[0 0 0])
axis square
set(gca,'visible','off')
saveas(gcf, '../figs/smile.png')