filePath = "D:\document\top\codes\End-to-End\Enter_End-to-End\LA\ASVspoof2019_LA_train\flac\LA_T_1000137.flac";
[x,fs] = audioread(filePath);
[stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
lfccs = [stat,delta,double_delta];
set(gca,'xtick',[],'xticklabel',[])
imagesc(lfccs')
axis xy;
colorbar;
xlabel('frames');
ylabel('lfcc dimension')
