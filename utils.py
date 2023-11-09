def train_auto(images, model, optimizer, loss_function, n_epochs=1):
    for _ in range(n_epochs):
        for image in images:

            reconstructed = model(image)
            loss = loss_function(reconstructed, image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

