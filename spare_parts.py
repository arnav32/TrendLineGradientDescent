
# plt.ion()
# # def plot_graph(x_vals, observeds, intercept, slope):
# fig = plt.figure(figsize=(5.5,5.5)) 

# ax = fig.add_subplot(111)
# # ax.set_ylim([0, max(observeds) * 1.2])
# ax.set_ylim(ymin=min(observeds) * 0.8, ymax=(max(observeds) * 1.2))








# plot_graph(x_vals, observeds, intercept, slope))
# print(squared_res(6, 2, 4, 8))
# print(deriv_squared_res_intercept(1.4, 0.5, 0.64))
# print(deriv_sum_sqared_res_intercept(x_vals, observeds, slope))








# -------- Draw updated graph --------
# plot_graph(x_vals, observeds, intercept, slope)
# x = np.linspace(min(x_vals), max(x_vals), 5) # takes: start, stop, number of vals
# m = slope
# c = intercept
# y = m*x+c


# line1, = ax.plot(x, y, 'r-')
# line1.set_ydata((slope * x) + intercept)

# plt.figure(figsize=(3.5,3.5))
# plt.plot(x, m*x+c, linestyle='solid')

# for i in range(len(x_vals)):
#     plt.plot(x_vals[i], observeds[i], 'bo')