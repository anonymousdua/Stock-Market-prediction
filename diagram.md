```mermaid
%%{init: {'theme': 'dark'}}%%
graph LR
    subgraph "1. UI Setup & User Input"
        A[Start: App Launches] --> B{Display Title & Sidebar};
        B --> C[/Get User Inputs<br/>- Stock Symbol<br/>- Lookback Period<br/>- Epochs & Batch Size<br/>- Prediction Days<br/>- Total Historical Days/];
        C --> D{Run Prediction Button Clicked?};
    end

    subgraph "2. Data Fetching & Preparation"
        D -- Yes --> E{Validate Parameters<br/>Total Days > Lookback + Prediction?};
        E -- No --> F[Show Error & Stop];
        E -- Yes --> G[Show Spinner: 'Fetching Data...'];
        G --> H(fetch_stock_data);
        H --> I{Data Fetched Successfully?};
        I -- No --> J[Show Error & Stop];
        I -- Yes --> K[Show Spinner: 'Adding Features...'];
        K --> L[Add Sentiment & SMAs];
        L --> M[Drop NaN Rows];
    end

    subgraph "3. Data Splitting & Scaling"
        M --> N[Show Spinner: 'Preparing Data...'];
        N --> O{Enough Data After<br/>Feature Engineering?};
        O -- No --> P[Show Error & Stop];
        O -- Yes --> Q[Split Data into<br/>Train & Test Sets];
        Q --> R[Scale Data with MinMaxScaler];
    end

    subgraph "4. Model Training"
        R --> S[Show Spinner: 'Creating Sequences...'];
        S --> T(create_sequences for Training);
        T --> U{Sufficient Training<br/>Sequences Created?};
        U -- No --> V[Show Error & Stop];
        U -- Yes --> W[Show Spinner: 'Building & Training Model...'];
        W --> X(build_lstm_model);
        X --> Y[Loop through Epochs];
        Y -- For each epoch --> Z["model.fit(X_train, y_train)"];
        Z --> Y_Update[Update Progress Bar & Status];
        Y_Update --> Y;
        Y -- Loop Finished --> AA[Show 'Training Completed' Message];
    end
    
    subgraph "5. Prediction & Evaluation"
        AA --> BB[Show Spinner: 'Making Predictions...'];
        BB --> CC(create_sequences for Testing);
        CC --> DD{Test Sequences Created?};
        DD -- No --> EE[Show Error & Stop];
        DD -- Yes --> FF["model.predict(X_test)"];
        FF --> GG[Inverse Transform Predictions];
        GG --> HH["Calculate Performance Metrics<br/>(RMSE, MAE, Accuracy, etc.)"];
    end

    subgraph "6. Display Results"
        HH --> II[Display Metrics in Columns];
        II --> JJ[Visualize Results:<br/>Plot Actual vs. Predicted Prices];
        JJ --> KK[Display Detailed Prediction Data Table];
    end

    D -- No --> D;
    F --> End((End));
    J --> End;
    P --> End;
    V --> End;
    EE --> End;
    KK --> End((End));

    style End fill:#963484,stroke:#fff,stroke-width:2px

'''

