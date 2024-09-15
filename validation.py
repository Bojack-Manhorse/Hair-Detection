import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def accuracy_score_from_valiadation(model, validation_loader):
    """
    Calculates the accuracy using the WHOLE of the validation dataset.
    """
    with torch.no_grad():
        losses = torch.zeros(0).to(device)

        for _, batch in enumerate(validation_loader):

            features, labels = batch

            model.train()

            features = list(img.to(device) for img in features)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            model.to(device)

            loss_dict = model(features, labels)
            loss = sum(loss for loss in loss_dict.values())

            losses = torch.cat((losses, loss.view(1)))

        accuracy_score = torch.sum(losses) / len(losses)

        return accuracy_score