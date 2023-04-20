def train(model, loader, optimizer):
    model.train()
    loss_all = 0
    for data in loader:
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(train_dataset)