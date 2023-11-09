def train_auto(images, model, optimizer, loss_function, n_epochs, device):
    for _ in range(n_epochs):
        for image in images:
            image = image.to(device)
            
            reconstructed = model(image)
            loss = loss_function(reconstructed, image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

