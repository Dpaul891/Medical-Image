import matplotlib.pyplot as plt
def pt(total_iter, name, lst, label):
    iterations = range(total_iter)
    plt.figure()
    plt.title(name)
    plt.plot(iterations, lst, label = label)
    plt.xlabel("iterations/100")
    plt.legend()
    plt.savefig(name+'.pdf')
    plt.show()

if __name__ == '__main__':
    # with open('dice.txt', 'r') as f:
    #     dice_list = eval(f.read())
    # pt(len(dice_list), 'validation dice coefficient', dice_list, 'validation dice coefficient')
    
    with open('loss.txt', 'r') as f:
        loss_list = f.read().splitlines()
    loss_list = [float(i) for i in loss_list]
    pt(len(loss_list), 'train loss', loss_list, 'train loss')

    
    
    
    
    
    