import threading

def loop(model, datagen, callback=None, epochs=1):
    for epoch in range(1,1+epochs):
        for itr,batch in enumerate(datagen,1):
            model.forwardbackward(*batch)

            step = itr + (len(datagen) * epoch - 1)
            if step % interval == 0:
                if callback: threading.thread(target=callback.__call__, args=datagen).start()
