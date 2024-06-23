import torch
import torch.nn.functional as F
from torch import nn

from utils.config import SHINEConfig


def get_geo_decoder(config: SHINEConfig):
    if config.decoder_type == "base":
        return Decoder(config, is_geo_encoder=True, is_time_conditioned=False, semantic_aware=False)
    elif config.decoder_type == "sea":
        return Decoder(config, is_geo_encoder=True, is_time_conditioned=False, semantic_aware=True)
    elif config.decoder_type == "sea_oh":
        return DecoderOH(config)
    elif config.decoder_type == "sea_emb":
        return DecoderEmb(config)
    elif config.decoder_type == "sea_film":
        return DecoderFilm(config)
    else:
        raise ValueError("Unknown decoder type: {}".format(config.decoder))


class Decoder(nn.Module):
    def __init__(
        self, config: SHINEConfig, is_geo_encoder=True, is_time_conditioned=False, semantic_aware=False
    ):
        super().__init__()

        if is_geo_encoder:
            mlp_hidden_dim = config.geo_mlp_hidden_dim
            mlp_bias_on = config.geo_mlp_bias_on
            mlp_level = config.geo_mlp_level
        else:
            mlp_hidden_dim = config.sem_mlp_hidden_dim
            mlp_bias_on = config.sem_mlp_bias_on
            mlp_level = config.sem_mlp_level

        self.semantic_aware = semantic_aware

        input_layer_count = config.feature_dim
        if is_time_conditioned:
            input_layer_count += 1
        if semantic_aware:
            input_layer_count += 1
        if config.use_pe:
            pos_enc_dim = config.pos_input_dim * (2 * config.pos_encoding_band + 1)
            input_layer_count += pos_enc_dim * len(config.pe_levels)
        if config.use_gaussian_pe:
            pos_enc_dim = config.pos_encoding_band * 2 + config.pos_input_dim
            input_layer_count += pos_enc_dim * len(config.pe_levels)

        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initializa the structure of shared MLP
        layers = []
        for i in range(mlp_level):
            if i == 0:
                layers.append(nn.Linear(input_layer_count, mlp_hidden_dim, mlp_bias_on))
            else:
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, mlp_bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(mlp_hidden_dim, 1, mlp_bias_on)
        self.nclass_out = nn.Linear(
            mlp_hidden_dim, config.sem_class_count + 1, mlp_bias_on
        )  # sem class + free space class

        self.to(config.device)

    def forward(self, feature, semantic_labels=None):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        if self.semantic_aware:
            output = self.sea_sdf(feature, semantic_labels)
        else:
            output = self.sdf(feature)
        return output

    # predict the sdf (opposite sign to the actual sdf)
    def sdf(self, sum_features):
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(sum_features))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out

    def sea_sdf(self, sum_features, semantic_labels):
        semantic_conditioned_feature = torch.torch.cat((sum_features, semantic_labels.view(-1, 1)), dim=1)

        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(semantic_conditioned_feature))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim + 1 -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out

    def time_conditionded_sdf(self, sum_features, ts):
        time_conditioned_feature = torch.torch.cat((sum_features, ts.view(-1, 1)), dim=1)

        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(time_conditioned_feature))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim + 1 -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out

    # predict the occupancy probability
    def occupancy(self, sum_features):
        out = torch.sigmoid(self.sdf(sum_features))  # to [0, 1]
        return out

    # predict the probabilty of each semantic label
    def sem_label_prob(self, sum_features):
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(sum_features))
            else:
                h = F.relu(l(h))

        out = F.log_softmax(self.nclass_out(h), dim=1)
        return out

    def sem_label(self, sum_features):
        out = torch.argmax(self.sem_label_prob(sum_features), dim=1)
        return out


class DecoderOH(nn.Module):
    def __init__(self, config: SHINEConfig):
        super().__init__()

        mlp_hidden_dim = config.geo_mlp_hidden_dim
        mlp_bias_on = config.geo_mlp_bias_on
        mlp_level = config.geo_mlp_level

        input_layer_count = config.feature_dim
        self.num_classes = config.sem_class_count + 1
        input_layer_count += self.num_classes

        if config.use_pe:
            pos_enc_dim = config.pos_input_dim * (2 * config.pos_encoding_band + 1)
            input_layer_count += pos_enc_dim * len(config.pe_levels)
        if config.use_gaussian_pe:
            pos_enc_dim = config.pos_encoding_band * 2 + config.pos_input_dim
            input_layer_count += pos_enc_dim * len(config.pe_levels)

        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initializa the structure of shared MLP
        layers = []
        for i in range(mlp_level):
            if i == 0:
                layers.append(nn.Linear(input_layer_count, mlp_hidden_dim, mlp_bias_on))
            else:
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, mlp_bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(mlp_hidden_dim, 1, mlp_bias_on)
        self.nclass_out = nn.Linear(
            mlp_hidden_dim, config.sem_class_count + 1, mlp_bias_on
        )  # sem class + free space class

        self.to(config.device)

    def sea_sdf(self, sum_features, semantic_labels):
        sl_one_hot = torch.zeros((semantic_labels.size(0), self.num_classes), device=semantic_labels.device)
        sl_one_hot.scatter_(1, semantic_labels.unsqueeze(1), 1)
        # Concatenate x and y_one_hot
        semantic_conditioned_feature = torch.cat((sum_features, sl_one_hot), dim=1)

        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(semantic_conditioned_feature))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)

        return out

    def forward(self, feature, semantic_labels):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.sea_sdf(feature, semantic_labels)
        return output


class DecoderEmb(nn.Module):
    def __init__(self, config: SHINEConfig):
        super().__init__()

        mlp_hidden_dim = config.geo_mlp_hidden_dim
        mlp_bias_on = config.geo_mlp_bias_on
        mlp_level = config.geo_mlp_level

        input_layer_count = config.feature_dim
        if config.use_pe:
            pos_enc_dim = config.pos_input_dim * (2 * config.pos_encoding_band + 1)
            input_layer_count += pos_enc_dim * len(config.pe_levels)
        if config.use_gaussian_pe:
            pos_enc_dim = config.pos_encoding_band * 2 + config.pos_input_dim
            input_layer_count += pos_enc_dim * len(config.pe_levels)
        self.num_classes = config.sem_class_count + 1

        self.embedding = nn.Embedding(self.num_classes, config.feature_dim)

        input_layer_count *= 2
        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initialize the structure of shared MLP
        layers = []
        for i in range(mlp_level):
            if i == 0:
                layers.append(nn.Linear(input_layer_count, mlp_hidden_dim, mlp_bias_on))
            else:
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, mlp_bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(mlp_hidden_dim, 1, mlp_bias_on)
        self.nclass_out = nn.Linear(
            mlp_hidden_dim, config.sem_class_count + 1, mlp_bias_on
        )  # sem class + free space class

        self.to(config.device)

    def sea_sdf(self, sum_features, semantic_labels):
        sl_emb = self.embedding(semantic_labels)
        # Concatenate x and y_one_hot
        semantic_conditioned_feature = torch.cat((sum_features, sl_emb), dim=1)

        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(semantic_conditioned_feature))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)

        return out

    def forward(self, feature, semantic_labels):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.sea_sdf(feature, semantic_labels)
        return output


class FiLMGenerator(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Simple MLP: Adjust architecture as needed
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size * 2),  # output_size should match the number of feature maps
        )

    def forward(self, x):
        # Generate gamma and beta
        film_params = self.net(x)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        return gamma, beta


class DecoderFilm(nn.Module):
    def __init__(self, config: SHINEConfig):
        super().__init__()

        mlp_hidden_dim = config.geo_mlp_hidden_dim
        mlp_bias_on = config.geo_mlp_bias_on
        mlp_level = config.geo_mlp_level

        input_layer_count = config.feature_dim
        if config.use_pe:
            pos_enc_dim = config.pos_input_dim * (2 * config.pos_encoding_band + 1)
            input_layer_count += pos_enc_dim * len(config.pe_levels)
        if config.use_gaussian_pe:
            pos_enc_dim = config.pos_encoding_band * 2 + config.pos_input_dim
            input_layer_count += pos_enc_dim * len(config.pe_levels)
        self.num_classes = config.sem_class_count + 1

        self.film_generator = FiLMGenerator(input_size=1, output_size=mlp_hidden_dim)

        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initialize the structure of shared MLP
        layers = []
        for i in range(mlp_level):
            if i == 0:
                layers.append(nn.Linear(input_layer_count, mlp_hidden_dim, mlp_bias_on))
            else:
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, mlp_bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(mlp_hidden_dim, 1, mlp_bias_on)
        self.nclass_out = nn.Linear(
            mlp_hidden_dim, config.sem_class_count + 1, mlp_bias_on
        )  # sem class + free space class

        self.to(config.device)

    def sea_sdf(self, sum_features, semantic_labels):
        gamma, beta = self.film_generator(semantic_labels.reshape(-1, 1).float())

        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(sum_features))
                h = gamma * h + beta
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)

        return out

    def forward(self, feature, semantic_labels):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.sea_sdf(feature, semantic_labels)
        return output
