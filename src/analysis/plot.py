################################################################################
def plot_history(history: "tf.keras.callback.History", save_to: "str"):

    for key, value in history.history.items():

        # fig, ax = plt.subplots(figsize=(10, 5))

        # ax.plot(value)

        # ax.set_title(f'Model {key}')
        # ax.set_xlabel(f'epochs')
        # ax.set_ylabel(f'{key}')

        # fig.savefig(f'{save_to}_{key}.png')
        # plt.close()

        np.save(f"{save_to}_{key}.npy", value)


################################################################################
def digits_plot():
    pass
