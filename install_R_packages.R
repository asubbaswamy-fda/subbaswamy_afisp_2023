load("R_packages.rda")

for (count in 1:length(installedpackages)) {
    install.packages(installedpackages[count])
}
