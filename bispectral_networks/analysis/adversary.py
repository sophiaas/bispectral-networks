import torch
import numpy as np


class BasicGradientDescent(torch.nn.Module):
    def __init__(
        self,
        model,
        target_image,
        initial_image=None,
        distance_fn=torch.nn.functional.pairwise_distance,
        margin=1.0,
        pnorm=2,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr=0.001,
        save_interval=100,
        print_interval=1000,
        device="cpu",
    ):

        super().__init__()

        self.device = device
        self.distance_fn = distance_fn
        self.pnorm = pnorm
        self.margin = margin

        self.save_interval = save_interval
        self.print_interval = print_interval

        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False

        self.target_image = target_image.to(device)
        
        self.target_embedding, _ = model(target_image.to(device))
        if len(self.target_embedding.shape) < 2:
            self.target_embedding = self.target_embedding.unsqueeze(0)
        self.target_embedding = self.target_embedding.detach()


        if initial_image is None:
            self.x_ = torch.nn.Parameter(torch.tensor(np.random.normal(loc=0, scale=1.0, size=target_image.shape)).to(device))
        else:
            self.x_ = torch.nn.Parameter(torch.tensor(initial_image)).to(device)
            
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.scheduler = scheduler(self.optimizer)

        self.history = []
        self.d_history = []

    def train(self, max_iter=1000):
        i = 0
        within_margin = False

        while not (within_margin or i == max_iter):
            embedding, _ = self.model(self.x_)
            if len(embedding.shape) < 2:
                embedding = embedding.unsqueeze(0)
            embedding = embedding.type(self.target_embedding.dtype)
            d = (self.distance_fn(self.target_embedding, embedding)).mean()
            
            if d < self.margin:
                within_margin = True

            else:
                self.optimizer.zero_grad()
                d.backward()
                self.optimizer.step()
                i += 1

                if i % self.save_interval == 0:
                    self.history.append(self.x_.cpu().detach().clone().numpy())
                    self.d_history.append(d.cpu().detach().numpy())

                if i % self.print_interval == 0:
                    print("Iter: {} | Distance: {}".format(i, d.detach()))
                    self.scheduler.step(d)
            
        print("Final Distance: {}".format(d))

        if not within_margin:
            print("Did not reach margin.")

        return self.x_.detach(), self.target_embedding.detach(), embedding.detach()
