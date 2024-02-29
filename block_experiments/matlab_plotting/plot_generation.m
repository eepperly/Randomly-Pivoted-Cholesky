load('../data/generation.mat')
close all

markers = {'o','*','s','^'};
colors = {'#648FFF','#785EF0','#DC267F','#FE6100'};

data = {gaussian, l1laplace};
names = {'$\ell_2$ Gaussian', '$\ell_1$ Laplace'};
fnames = {'generation_gaussian','generation_l1laplace'};

mymax = -Inf;
mymin = Inf;

for data_idx = 1:length(data)
    results = data{data_idx};
    name = names{data_idx};
    fname = fnames{data_idx};
    figure(data_idx);

    ds = unique(results(:,1));
    legend_entries = cell(length(ds),1);
    for d_idx = 1:length(ds)
        d = ds(d_idx);
        myresults = results(results(:,1)==d,2:3);
        loglog(myresults(:,1), 1 ./ myresults(:,2),'Marker',...
            markers{d_idx},'MarkerFaceColor',colors{d_idx},...
            'Color',colors{d_idx},'LineWidth',2,...
            'MarkerSize',10)
        hold on
        legend_entries{d_idx} = sprintf('$d = %d$',d);
        mymax = max(mymax, max(1 ./ myresults(:,2)));
        mymin = min(mymin, min(1 ./ myresults(:,2)));
    end

    xlabel('Block size $b$')
    ylabel('Columns generated per second')
    if data_idx == 1
        legend(legend_entries, 'Location', 'southeast')
    end
end

for data_idx = 1:length(data)
    figure(data_idx)
    axis([-Inf Inf 0.8*mymin 1.2*mymax])
    fname = fnames{data_idx};

    saveas(gcf, sprintf('../figs/%s.png',fname))
    saveas(gcf, sprintf('../figs/%s.fig',fname))
end